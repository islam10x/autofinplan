import asyncio
import threading
import time
import sys
from typing import Dict, Any, Optional, Callable
from Agents.Prortfolio_agent import RLPortfolioAgent
from Agents.dept_agent import RLDebtAgent

# Try to import rich for better terminal display, fallback to basic if not available
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available. Using basic progress display. Install with: pip install rich")

class RLTrainingService:
    """
    Enhanced service for training and managing RL agents with stop control and progress tracking
    """
    
    def __init__(self):
        self.agents = {
            'portfolio': RLPortfolioAgent("PPO"),
            'debt': RLDebtAgent()
        }
        
        # Training control
        self.stop_event = threading.Event()
        self.training_lock = threading.Lock()
        
        # Progress display
        self.console = Console() if RICH_AVAILABLE else None
        self.progress_display = None
        self.live_display = None
        self.progress_tasks = {}
        
        # Training state
        self.training_state = {
            "is_training": False,
            "current_agent": None,
            "current_timestep": 0,
            "total_timesteps": 0,
            "agent_progress": {},
            "start_time": None,
            "training_id": None,
            "stop_requested": False,
            "progress": 0.0,
            "status": "idle",  # idle, training, stopping, completed, error
            "current_agent_index": 0,
            "total_agents": len(self.agents)
        }
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state (thread-safe)"""
        with self.training_lock:
            return self.training_state.copy()
        
    def _update_training_state(self, updates: Dict[str, Any]):
        """Thread-safe state update"""
        with self.training_lock:
            self.training_state.update(updates)
    
    def _setup_progress_display(self):
        """Setup rich progress display"""
        if not RICH_AVAILABLE:
            return
            
        self.progress_display = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} timesteps"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            refresh_per_second=2,
        )
        
        # Add tasks for each agent
        self.progress_tasks = {}
        for agent_name in self.agents.keys():
            task_id = self.progress_display.add_task(
                f"Training {agent_name.title()} Agent",
                total=100,
                visible=False
            )
            self.progress_tasks[agent_name] = task_id
        
        # Add overall progress task
        self.progress_tasks['overall'] = self.progress_display.add_task(
            "Overall Training Progress",
            total=100,
            visible=True
        )
    
    def _create_info_panel(self):
        """Create info panel for training status"""
        if not RICH_AVAILABLE:
            return None
            
        status = self.get_training_state()
        
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Label", style="bold cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Training ID:", status.get("training_id", "N/A"))
        info_table.add_row("Status:", status.get("status", "Unknown").title())
        info_table.add_row("Current Agent:", status.get("current_agent", "None").title() if status.get("current_agent") else "None")
        
        if status.get("start_time"):
            elapsed = time.time() - status["start_time"]
            info_table.add_row("Elapsed Time:", f"{elapsed:.1f}s")
        
        if status.get("total_timesteps"):
            info_table.add_row("Total Timesteps:", f"{status['total_timesteps']:,}")
            
        if status.get("current_timestep"):
            info_table.add_row("Current Timestep:", f"{status['current_timestep']:,}")
        
        return Panel(info_table, title="Training Information", border_style="green")
    
    def _update_progress_display(self, agent_name: str, progress: float, timestep: int = 0):
        """Update progress display"""
        if not RICH_AVAILABLE or not self.progress_display:
            # Fallback to simple print
            print(f"\r{agent_name.title()} Agent: {progress:.1f}% ({timestep:,} timesteps)", end="", flush=True)
            return
            
        # Update agent-specific progress
        if agent_name in self.progress_tasks:
            self.progress_display.update(
                self.progress_tasks[agent_name],
                completed=progress,
                visible=True
            )
        
        # Update overall progress
        overall_progress = self.training_state.get("progress", 0)
        self.progress_display.update(
            self.progress_tasks['overall'],
            completed=overall_progress
        )
    
    def _start_live_display(self):
        """Start live display"""
        if not RICH_AVAILABLE:
            print("\n" + "="*60)
            print("🚀 STARTING RL AGENT TRAINING")
            print("="*60)
            return
            
        self._setup_progress_display()
        
        def create_layout():
            info_panel = self._create_info_panel()
            if info_panel:
                return Panel.fit(
                    self.progress_display,
                    title="🤖 RL Agent Training Progress",
                    border_style="blue"
                )
            return self.progress_display
        
        self.live_display = Live(
            create_layout(),
            refresh_per_second=2,
            console=self.console
        )
        self.live_display.start()
    
    def _stop_live_display(self):
        """Stop live display"""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
        
        if not RICH_AVAILABLE:
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETED")
            print("="*60)
        else:
            self.console.print("\n✅ [bold green]Training Session Completed![/bold green]")
    
    def _simple_progress_print(self, agent_name: str, progress: float, timestep: int, total_timesteps: int):
        """Simple progress printing for when rich is not available"""
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r{agent_name.title()}: |{bar}| {progress:.1f}% ({timestep:,}/{total_timesteps:,})", end="", flush=True)
        """Get current training state (thread-safe)"""
        with self.training_lock:
            return self.training_state.copy()
    
    def _create_progress_callback(self, agent_name: str, agent_index: int) -> Callable:
        """Create a progress callback for an agent"""
        def progress_callback(locals_dict, globals_dict):
            """Callback function for training progress"""
            if self.stop_event.is_set():
                return False  # Stop training
            
            try:
                        # Extract current timestep from the training locals
                        current_timestep = locals_dict.get('self', {}).get('num_timesteps', 0)
                        
                        # Update state
                        agent_progress = (current_timestep / self.training_state["total_timesteps"]) * 100
                        overall_progress = ((agent_index * 100) + agent_progress) / self.training_state["total_agents"]
                        
                        self._update_training_state({
                            "current_timestep": current_timestep,
                            "progress": overall_progress,
                            "agent_progress": {
                                **self.training_state["agent_progress"],
                                agent_name: agent_progress
                            }
                        })
                        
                        # Update live display
                        self._update_progress_display(agent_name, agent_progress, current_timestep)
                        
            except Exception as e:
                        # Continue training even if progress update fails
                        print(f"Progress callback error for {agent_name}: {e}")
            
            return True  # Continue training
        
        return progress_callback
    
    async def train_all_agents(self, timesteps: int = 100000, training_id: str = None):
        """Train all RL agents with stop control and progress tracking"""
        
        # Initialize training state
        training_id = training_id or f"train_{int(time.time())}"
        self._update_training_state({
            "is_training": True,
            "current_agent": None,
            "current_timestep": 0,
            "total_timesteps": timesteps,
            "agent_progress": {},
            "start_time": time.time(),
            "training_id": training_id,
            "stop_requested": False,
            "progress": 0.0,
            "status": "training",
            "current_agent_index": 0,
            "total_agents": len(self.agents)
        })
        
        # Clear stop event
        self.stop_event.clear()
        
        # Start live progress display
        self._start_live_display()
        
        try:
            agent_names = list(self.agents.keys())
            
            for agent_index, (name, agent) in enumerate(self.agents.items()):
                # Check if stop was requested
                if self.stop_event.is_set():
                    self._update_training_state({"status": "stopping"})
                    break
                
                print(f"Training {name} agent... ({agent_index + 1}/{len(self.agents)})")
                
                self._update_training_state({
                    "current_agent": name,
                    "current_agent_index": agent_index,
                    "current_timestep": 0
                })
                
                try:
                    # Try different training approaches in order of preference
                    training_completed = False
                    
                    # Approach 1: Try with progress callback (stable-baselines3 style)
                    try:
                        progress_callback = self._create_progress_callback(name, agent_index)
                        
                        # Try to train with callback parameter
                        await asyncio.get_event_loop().run_in_executor(
                            None, 
                            self._train_agent_with_callback,
                            agent,
                            timesteps,
                            progress_callback
                        )
                        training_completed = True
                        
                    except Exception as callback_error:
                        print(f"Callback training failed for {name}: {callback_error}")
                        
                        # Approach 2: Check if agent has custom train_with_callback method
                        if hasattr(agent, 'train_with_callback'):
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, 
                                    agent.train_with_callback, 
                                    timesteps, 
                                    progress_callback,
                                    self.stop_event
                                )
                                training_completed = True
                            except Exception as custom_error:
                                print(f"Custom callback training failed for {name}: {custom_error}")
                    
                    # Approach 3: Fallback to regular training with progress estimation
                    if not training_completed:
                        print(f"Using fallback training method for {name} agent...")
                        
                        training_task = asyncio.get_event_loop().run_in_executor(
                            None, 
                            agent.train, 
                            timesteps
                        )
                        
                        # Monitor progress periodically with better estimation
                        start_time = time.time()
                        while not training_task.done():
                            if self.stop_event.is_set():
                                training_task.cancel()
                                break
                            
                            await asyncio.sleep(2)  # Check every 2 seconds
                            
                            # Better progress estimation based on typical training times
                            elapsed = time.time() - start_time
                            
                            # Estimate based on timesteps (rough estimate: 1000 timesteps per second)
                            estimated_completion_time = timesteps / 1000  # seconds
                            agent_progress = min(95, (elapsed / estimated_completion_time) * 100)
                            overall_progress = ((agent_index * 100) + agent_progress) / len(self.agents)
                            
                            # Update state
                            self._update_training_state({
                                "current_timestep": int(agent_progress * timesteps / 100),
                                "progress": overall_progress,
                                "agent_progress": {
                                    **self.training_state["agent_progress"],
                                    name: agent_progress
                                }
                            })
                            
                            # Update display
                            current_timestep = int(agent_progress * timesteps / 100)
                            self._update_progress_display(name, agent_progress, current_timestep)
                            
                            # Also show simple progress if rich not available
                            if not RICH_AVAILABLE:
                                self._simple_progress_print(name, agent_progress, current_timestep, timesteps)
                        
                        # Wait for completion
                        if not self.stop_event.is_set():
                            await training_task
                    
                    if not self.stop_event.is_set():
                        print(f"\n✅ {name} agent training completed!")
                        # Mark this agent as 100% complete
                        self._update_training_state({
                            "agent_progress": {
                                **self.training_state["agent_progress"],
                                name: 100.0
                            }
                        })
                        # Update display to show completion
                        self._update_progress_display(name, 100.0, timesteps)
                    else:
                        print(f"\n🛑 {name} agent training stopped!")
                        break
                        
                except Exception as e:
                    print(f"Error training {name} agent: {e}")
                    self._update_training_state({
                        "status": "error",
                        "error_message": f"Error training {name} agent: {str(e)}"
                    })
                    raise
            
            # Training completed or stopped
            if not self.stop_event.is_set():
                self._update_training_state({
                    "status": "completed",
                    "progress": 100.0,
                    "current_agent": None
                })
                if RICH_AVAILABLE and self.console:
                    self.console.print("\n🎉 [bold green]All agents training completed![/bold green]")
                else:
                    print("\n🎉 All agents training completed!")
            else:
                self._update_training_state({
                    "status": "stopped",
                    "current_agent": None
                })
                if RICH_AVAILABLE and self.console:
                    self.console.print("\n🛑 [bold yellow]Training stopped by user request![/bold yellow]")
                else:
                    print("\n🛑 Training stopped by user request!")
                
        except Exception as e:
            self._update_training_state({
                "status": "error",
                "error_message": str(e),
                "current_agent": None
            })
            if RICH_AVAILABLE and self.console:
                self.console.print(f"\n❌ [bold red]Training failed with error: {e}[/bold red]")
            else:
                print(f"\n❌ Training failed with error: {e}")
            raise
        finally:
            self._update_training_state({"is_training": False})
            self._stop_live_display()
    
    def stop_training(self):
        """Stop the currently running training"""
        if not self.training_state["is_training"]:
            return False, "No training currently running"
        
        self._update_training_state({
            "stop_requested": True,
            "status": "stopping"
        })
        
        self.stop_event.set()
        return True, "Stop signal sent to training process"
    
    def reset_training_state(self):
        """Reset training state (use when training is stuck)"""
        with self.training_lock:
            if self.training_state["is_training"] and self.training_state["status"] == "training":
                return False, "Cannot reset while training is active. Stop training first."
            
            self.training_state.update({
                "is_training": False,
                "current_agent": None,
                "current_timestep": 0,
                "total_timesteps": 0,
                "agent_progress": {},
                "start_time": None,
                "training_id": None,
                "stop_requested": False,
                "progress": 0.0,
                "status": "idle",
                "current_agent_index": 0,
                "total_agents": len(self.agents)
            })
        
        self.stop_event.clear()
        return True, "Training state reset successfully"
    
    def save_models(self, base_path: str = "./trained_models/"):
        """Save all trained models"""
        saved_models = []
        for name, agent in self.agents.items():
            try:
                model_path = f"{base_path}{name}_model"
                agent.model.save(model_path)
                saved_models.append(name)
                print(f"Saved {name} model to {model_path}")
            except Exception as e:
                print(f"Failed to save {name} model: {e}")
        
        return saved_models
    
    def load_models(self, base_path: str = "./trained_models/"):
        """Load pre-trained models"""
        loaded_models = []
        for name, agent in self.agents.items():
            model_path = f"{base_path}{name}_model"
            try:
                agent.load_model(model_path)
                loaded_models.append(name)
                print(f"Loaded {name} model from {model_path}")
            except Exception as e:
                print(f"Failed to load {name} model: {e}")
        
    def _train_agent_with_callback(self, agent, timesteps: int, callback):
        """Helper method to train agent with callback"""
        # Try different ways to add callback to training
        try:
            # Method 1: Standard stable-baselines3 callback
            if hasattr(agent, 'model') and hasattr(agent.model, 'learn'):
                agent.model.learn(total_timesteps=timesteps, callback=callback)
            elif hasattr(agent, 'train'):
                # Method 2: Pass callback to train method if supported
                agent.train(total_timesteps=timesteps, callback=callback)
            else:
                raise AttributeError("No suitable training method found")
        except TypeError as e:
            if "callback" in str(e):
                # Callback not supported, fall back to regular training
                raise Exception(f"Callback not supported: {e}")
            else:
                raise

    def get_agent_info(self):
        """Get information about available agents"""
        return {
            "agents": list(self.agents.keys()),
            "total_agents": len(self.agents),
            "agent_types": {name: type(agent).__name__ for name, agent in self.agents.items()}
        }