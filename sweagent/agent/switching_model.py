"""
Model wrapper that can switch between multiple models during execution.

This module provides a SwitchingModel class that wraps multiple models and
switches between them based on configurable criteria like step count,
total tokens, or custom logic.
"""

from __future__ import annotations

import random
from typing import Any
from sweagent.agent.models import (
    AbstractModel, 
    InstanceStats,
    SwitchingModelConfig,
    ModelAgentConfig,
    ToolConfig,
    get_model,
)
from sweagent.types import History
from sweagent.utils.log import get_logger



class SwitchingModel(AbstractModel):
    """Model that can switch between multiple models based on criteria"""
    
    def __init__(self, config: SwitchingModelConfig, tools: ToolConfig):
        self.config = config
        self.tools = tools
        self.logger = get_logger("swea-switching-model", emoji="🔄")
        
        # Store model configurations with their agent configs
        self.model_configs = config.model_configs or []
        if not self.model_configs:
            raise ValueError("SwitchingModel requires at least one model configuration")
        
        # Initialize all models
        self.models: list[AbstractModel] = []
        for model_agent_config in self.model_configs:
            model = get_model(model_agent_config.model, tools)
            self.models.append(model)
        
        # State tracking
        self.current_model_idx = 0
        self.step_count = 0
        self.switch_points: list[dict[str, Any]] = []  # Track when switches occurred
        
        # Combined stats across all models
        self.stats = InstanceStats()
        
        # For random switching - store the randomly selected switch step for each criteria
        self._random_switch_steps: dict[int, int | None] = {}
        
        # Callback for notifying agent of configuration changes
        self._config_change_callback: callable | None = None
        
        # Get model names safely
        model_names = []
        for m in self.models:
            if hasattr(m, 'config') and hasattr(m.config, 'name'):
                model_names.append(m.config.name)
            elif hasattr(m, 'name'):
                model_names.append(m.name)
            else:
                model_names.append(f"model_{len(model_names)}")
        
        self.logger.info(
            f"Initialized SwitchingModel with {len(self.models)} models: {model_names}"
        )
    
    @property
    def current_model(self) -> AbstractModel:
        """Get the currently active model"""
        return self.models[self.current_model_idx]
    
    @property
    def current_model_name(self) -> str:
        """Get the name of the currently active model"""
        model = self.current_model
        if hasattr(model, 'config') and hasattr(model.config, 'name'):
            return model.config.name
        elif hasattr(model, 'name'):
            return model.name
        else:
            return f"model_{self.current_model_idx}"
    
    @property 
    def instance_cost_limit(self) -> float:
        """Cost limit for the model"""
        return self.config.per_instance_cost_limit
    
    @property
    def current_agent_config(self) -> ModelAgentConfig:
        """Get the current model's agent configuration"""
        return self.model_configs[self.current_model_idx]
    
    def set_config_change_callback(self, callback: callable) -> None:
        """Set callback to be called when agent configuration should change"""
        self._config_change_callback = callback
    
    def _should_switch(self, criteria_idx: int) -> bool:
        """Check if we should switch to the next model based on criteria"""
        if criteria_idx >= len(self.config.switching_criteria):
            return False
            
        criteria = self.config.switching_criteria[criteria_idx]
        
        if criteria.type == "step_count":
            return self.step_count >= criteria.threshold
        
        elif criteria.type == "token_count":
            total_tokens = self.stats.tokens_sent + self.stats.tokens_received
            return total_tokens >= criteria.threshold
        
        elif criteria.type == "cost":
            return self.stats.instance_cost >= criteria.threshold
        
        elif criteria.type == "random_step_range":
            # Handle random switching within a step range
            if criteria.start_step is None or criteria.end_step is None:
                self.logger.error("random_step_range requires start_step and end_step")
                return False
            
            # Initialize random switch step if not set
            if criteria_idx not in self._random_switch_steps:
                self._random_switch_steps[criteria_idx] = random.randint(
                    criteria.start_step, criteria.end_step
                )
                self.logger.debug(
                    f"Random switch step selected: {self._random_switch_steps[criteria_idx]} "
                    f"(range: {criteria.start_step}-{criteria.end_step})"
                )
            
            # Check if we've reached the random switch step
            return self.step_count >= self._random_switch_steps[criteria_idx]
        
        elif criteria.type == "custom" and criteria.custom_function:
            # Execute custom function
            try:
                # Create a safe namespace for execution
                namespace = {
                    "model_idx": self.current_model_idx,
                    "stats": self.stats,
                    "step_count": self.step_count,
                }
                exec(f"result = {criteria.custom_function}", namespace)
                return bool(namespace.get("result", False))
            except Exception as e:
                self.logger.error(f"Error executing custom switching function: {e}")
                return False
        
        return False
    
    def _switch_to_next_model(self) -> None:
        """Switch to the next model in the list"""
        if self.current_model_idx + 1 >= len(self.models):
            self.logger.warning("No more models to switch to")
            return
        
        old_model_name = self.current_model_name
        old_config = self.current_agent_config
        self.current_model_idx += 1
        new_model_name = self.current_model_name
        new_config = self.current_agent_config
        
        # Record switch point
        self.switch_points.append({
            "step": self.step_count,
            "from_model": old_model_name,
            "to_model": new_model_name,
            "stats_at_switch": self.stats.model_dump(),
        })
        
        self.logger.info(
            f"🔄 Switching from {old_model_name} to {new_model_name} "
            f"at step {self.step_count} (cost: ${self.stats.instance_cost:.3f}, "
            f"tokens: {self.stats.tokens_sent + self.stats.tokens_received})"
        )
        
        # Notify agent of configuration change if callback is set
        if self._config_change_callback and self._has_config_changes(old_config, new_config):
            self.logger.info(f"🔧 Applying agent configuration changes for {new_model_name}")
            new_history = self._config_change_callback(new_config)
            return new_history
    
    def _has_config_changes(self, old_config: ModelAgentConfig, new_config: ModelAgentConfig) -> bool:
        """Check if there are any agent configuration changes between models"""
        return (
            new_config.templates is not None
            or new_config.tools is not None
            or new_config.history_processors is not None
        )
    
    def query(self, history: History, **kwargs) -> dict | list[dict]:
        """Query the current model, switching if criteria are met"""
        
        # Check if we should switch models
        if self.current_model_idx < len(self.models) - 1:
            if self._should_switch(self.current_model_idx):
                new_history = self._switch_to_next_model()
                if new_history is not None:
                    history = new_history
        
        # Query current model
        result = self.current_model.query(history, **kwargs)
        
        # Update our combined stats
        self.stats = InstanceStats(
            instance_cost=sum(m.stats.instance_cost for m in self.models),
            tokens_sent=sum(m.stats.tokens_sent for m in self.models),
            tokens_received=sum(m.stats.tokens_received for m in self.models),
            api_calls=sum(m.stats.api_calls for m in self.models),
        )
        
        # Increment step count
        self.step_count += 1
        
        # Log current state periodically
        if self.step_count % 5 == 0:
            self.logger.debug(
                f"Step {self.step_count}: Using {self.current_model_name}, "
                f"Cost: ${self.stats.instance_cost:.3f}, "
                f"Tokens: {self.stats.tokens_sent + self.stats.tokens_received}"
            )
        
        return result
    
    def reset_stats(self) -> None:
        """Reset stats for all models"""
        for model in self.models:
            model.reset_stats()
        self.stats = InstanceStats()
        self.step_count = 0
        self.current_model_idx = 0
        self.switch_points = []
        self._random_switch_steps = {}
    
    def get_info(self) -> dict[str, Any]:
        """Get information about the switching model's state"""
        # Get model names safely
        model_names = []
        for i, m in enumerate(self.models):
            if hasattr(m, 'config') and hasattr(m.config, 'name'):
                model_names.append(m.config.name)
            elif hasattr(m, 'name'):
                model_names.append(m.name)
            else:
                model_names.append(f"model_{i}")
        
        # Create individual stats dict with safe names
        individual_stats = {}
        for i, m in enumerate(self.models):
            name = model_names[i]
            if hasattr(m, 'stats'):
                individual_stats[name] = m.stats.model_dump()
            else:
                individual_stats[name] = {}
        
        return {
            "current_model": self.current_model_name,
            "current_model_idx": self.current_model_idx,
            "step_count": self.step_count,
            "switch_points": self.switch_points,
            "models": model_names,
            "stats": self.stats.model_dump(),
            "individual_model_stats": individual_stats
        }
