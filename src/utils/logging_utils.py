"""
Logging utilities for maritime anomaly detection system.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add color to levelname
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_name: str = "maritime_anomaly_detection",
    use_wandb: bool = False
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("maritime_anomaly_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    # Weights & Biases integration
    if use_wandb:
        try:
            import wandb
            # wandb.init will be called in the training script
            logger.info("Weights & Biases logging enabled")
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")
    
    return logger


def log_config(config: dict, logger: logging.Logger) -> None:
    """
    Log configuration parameters.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration:")
    logger.info("-" * 50)
    
    def _log_dict(d: dict, indent: int = 0) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                _log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    _log_dict(config)
    logger.info("-" * 50)


def log_model_summary(model, logger: logging.Logger) -> None:
    """
    Log model architecture summary.
    
    Args:
        model: Model instance
        logger: Logger instance
    """
    logger.info("Model Architecture:")
    logger.info("-" * 50)
    
    # Count parameters
    if hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model structure
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Model structure:\n{str(model)}")
    logger.info("-" * 50)


def log_metrics(metrics: dict, epoch: int, logger: logging.Logger, prefix: str = "") -> None:
    """
    Log training/validation metrics.
    
    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch number
        logger: Logger instance
        prefix: Prefix for log messages (e.g., "Train", "Val")
    """
    prefix_str = f"{prefix} " if prefix else ""
    logger.info(f"{prefix_str}Epoch {epoch} Metrics:")
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")


def log_experiment_start(config: dict, logger: logging.Logger) -> None:
    """
    Log experiment start information.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("MARITIME ANOMALY DETECTION EXPERIMENT STARTED")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiment: {config.get('logging', {}).get('experiment_name', 'Unknown')}")
    log_config(config, logger)


def log_experiment_end(results: dict, logger: logging.Logger) -> None:
    """
    Log experiment end information.
    
    Args:
        results: Final results dictionary
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results:
        logger.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 80)


class ExperimentLogger:
    """
    Experiment logger class for managing logging throughout the experiment.
    """
    
    def __init__(self, config: dict):
        """
        Initialize experiment logger.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logging_config = config.get('logging', {})
        
        self.logger = setup_logging(
            log_level=logging_config.get('level', 'INFO'),
            log_dir=logging_config.get('log_dir'),
            experiment_name=logging_config.get('experiment_name', 'maritime_anomaly_detection'),
            use_wandb=logging_config.get('use_wandb', False)
        )
        
        # Initialize wandb if enabled
        if logging_config.get('use_wandb', False):
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            wandb.init(
                project="maritime-anomaly-detection",
                name=self.config.get('logging', {}).get('experiment_name'),
                config=self.config
            )
            self.logger.info("Weights & Biases initialized successfully")
        except ImportError:
            self.logger.warning("wandb not installed. Skipping W&B initialization.")
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
    
    def log_start(self) -> None:
        """Log experiment start."""
        log_experiment_start(self.config, self.logger)
    
    def log_end(self, results: dict) -> None:
        """Log experiment end."""
        log_experiment_end(results, self.logger)
    
    def log_metrics(self, metrics: dict, epoch: int, prefix: str = "") -> None:
        """Log metrics."""
        log_metrics(metrics, epoch, self.logger, prefix)
        
        # Log to wandb if available
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()}
                wandb_metrics['epoch'] = epoch
                wandb.log(wandb_metrics)
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def log_model(self, model) -> None:
        """Log model summary."""
        log_model_summary(model, self.logger)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message) 