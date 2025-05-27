#!/usr/bin/env python3
"""
Maritime Anomaly Detection Competition Runner

This script provides a unified interface to run different models:
1. TrAISformer (Deep Learning)
2. Traditional ML Models
3. Advanced Ensemble (TrAISformer + ML)

Usage:
    python run_competition.py --model traisformer --data data/ --output outputs/
    python run_competition.py --model ensemble --data data/ --output outputs/ --cv
    python run_competition.py --model all --data data/ --output outputs/ --submission
"""

import argparse
import logging
import yaml
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompetitionRunner:
    """
    Main runner for the maritime anomaly detection competition.
    """
    
    def __init__(self, args):
        """
        Initialize the competition runner.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.configs = {
            'traisformer': 'config/traisformer_config.yaml',
            'ensemble': 'config/ensemble_config.yaml'
        }
        
        # Training scripts
        self.scripts = {
            'traisformer': 'src/training/train_traisformer.py',
            'ensemble': 'src/training/train_ensemble.py'
        }
        
        self.results = {}
    
    def run_traisformer(self) -> Dict:
        """
        Run TrAISformer model training and evaluation.
        
        Returns:
            Results dictionary
        """
        logger.info("=" * 60)
        logger.info("RUNNING TRAISFORMER MODEL")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Prepare command
        cmd = [
            sys.executable,
            self.scripts['traisformer'],
            '--config', self.configs['traisformer'],
            '--data', self.args.data,
            '--output', str(self.output_dir / 'traisformer')
        ]
        
        try:
            # Run training
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.info("TrAISformer Output:")
                logger.info(result.stdout)
            
            # Load results
            results_path = self.output_dir / 'traisformer' / 'results.yaml'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    traisformer_results = yaml.safe_load(f)
                self.results['traisformer'] = traisformer_results
            
            elapsed_time = time.time() - start_time
            logger.info(f"TrAISformer completed in {elapsed_time:.2f} seconds")
            
            return self.results.get('traisformer', {})
            
        except subprocess.CalledProcessError as e:
            logger.error(f"TrAISformer training failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return {}
    
    def run_ensemble(self) -> Dict:
        """
        Run ensemble model training and evaluation.
        
        Returns:
            Results dictionary
        """
        logger.info("=" * 60)
        logger.info("RUNNING ADVANCED ENSEMBLE MODEL")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Prepare command
        cmd = [
            sys.executable,
            self.scripts['ensemble'],
            '--config', self.configs['ensemble'],
            '--data', self.args.data,
            '--output', str(self.output_dir / 'ensemble')
        ]
        
        # Add optional flags
        if self.args.cv:
            cmd.append('--cv')
        if self.args.submission:
            cmd.append('--submission')
        
        try:
            # Run training
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.info("Ensemble Output:")
                logger.info(result.stdout)
            
            # Load results
            results_path = self.output_dir / 'ensemble' / 'results.yaml'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    ensemble_results = yaml.safe_load(f)
                self.results['ensemble'] = ensemble_results
            
            elapsed_time = time.time() - start_time
            logger.info(f"Ensemble completed in {elapsed_time:.2f} seconds")
            
            return self.results.get('ensemble', {})
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ensemble training failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return {}
    
    def run_traditional_ml(self) -> Dict:
        """
        Run traditional ML models (using existing ensemble script without TrAISformer).
        
        Returns:
            Results dictionary
        """
        logger.info("=" * 60)
        logger.info("RUNNING TRADITIONAL ML MODELS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Use existing ensemble model script
        cmd = [
            sys.executable,
            'main.py',
            '--model', 'ensemble',
            '--data', self.args.data,
            '--output', str(self.output_dir / 'traditional_ml')
        ]
        
        try:
            # Run training
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                logger.info("Traditional ML Output:")
                logger.info(result.stdout)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Traditional ML completed in {elapsed_time:.2f} seconds")
            
            return {}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Traditional ML training failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            return {}
    
    def compare_models(self) -> None:
        """
        Compare results from different models.
        """
        if not self.results:
            logger.warning("No results available for comparison")
            return
        
        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        # Extract test metrics
        comparison_data = {}
        
        for model_name, results in self.results.items():
            if 'test_metrics' in results:
                comparison_data[model_name] = results['test_metrics']
            elif 'metrics' in results:
                comparison_data[model_name] = results['metrics']
        
        if not comparison_data:
            logger.warning("No test metrics found for comparison")
            return
        
        # Print comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Header
        header = f"{'Model':<15}"
        for metric in metrics:
            header += f"{metric.upper():<12}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Results
        for model_name, model_metrics in comparison_data.items():
            row = f"{model_name:<15}"
            for metric in metrics:
                value = model_metrics.get(metric, 0.0)
                row += f"{value:<12.4f}"
            logger.info(row)
        
        # Find best model for each metric
        logger.info("\nBest Models by Metric:")
        for metric in metrics:
            best_model = max(comparison_data.keys(), 
                           key=lambda x: comparison_data[x].get(metric, 0.0))
            best_score = comparison_data[best_model].get(metric, 0.0)
            logger.info(f"  {metric.upper()}: {best_model} ({best_score:.4f})")
    
    def generate_final_submission(self) -> None:
        """
        Generate final submission using the best performing model.
        """
        if not self.results:
            logger.warning("No results available for submission generation")
            return
        
        logger.info("=" * 60)
        logger.info("GENERATING FINAL SUBMISSION")
        logger.info("=" * 60)
        
        # Find best model based on AUC score
        best_model = None
        best_auc = 0.0
        
        for model_name, results in self.results.items():
            if 'test_metrics' in results:
                auc = results['test_metrics'].get('roc_auc', 0.0)
            elif 'metrics' in results:
                auc = results['metrics'].get('roc_auc', 0.0)
            else:
                continue
            
            if auc > best_auc:
                best_auc = auc
                best_model = model_name
        
        if best_model:
            logger.info(f"Best model: {best_model} (AUC: {best_auc:.4f})")
            
            # Copy submission file
            submission_source = self.output_dir / best_model / 'submission.csv'
            submission_dest = self.output_dir / 'final_submission.csv'
            
            if submission_source.exists():
                import shutil
                shutil.copy2(submission_source, submission_dest)
                logger.info(f"Final submission saved to: {submission_dest}")
            else:
                logger.warning(f"Submission file not found for {best_model}")
        else:
            logger.warning("No valid model found for submission generation")
    
    def save_summary(self) -> None:
        """
        Save a summary of all results.
        """
        summary = {
            'experiment_info': {
                'data_path': self.args.data,
                'output_path': str(self.output_dir),
                'models_run': list(self.results.keys()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': self.results
        }
        
        summary_path = self.output_dir / 'experiment_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Experiment summary saved to: {summary_path}")
    
    def run(self) -> None:
        """
        Main execution method.
        """
        logger.info("Starting Maritime Anomaly Detection Competition")
        logger.info(f"Data path: {self.args.data}")
        logger.info(f"Output path: {self.output_dir}")
        logger.info(f"Models to run: {self.args.model}")
        
        start_time = time.time()
        
        # Run specified models
        if self.args.model in ['traisformer', 'all']:
            self.run_traisformer()
        
        if self.args.model in ['traditional', 'all']:
            self.run_traditional_ml()
        
        if self.args.model in ['ensemble', 'all']:
            self.run_ensemble()
        
        # Compare models if multiple were run
        if len(self.results) > 1:
            self.compare_models()
        
        # Generate final submission
        if self.args.submission:
            self.generate_final_submission()
        
        # Save summary
        self.save_summary()
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("COMPETITION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Maritime Anomaly Detection Competition Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TrAISformer only
  python run_competition.py --model traisformer --data data/ --output outputs/

  # Run ensemble with cross-validation
  python run_competition.py --model ensemble --data data/ --output outputs/ --cv

  # Run all models and generate submission
  python run_competition.py --model all --data data/ --output outputs/ --submission

  # Run traditional ML models only
  python run_competition.py --model traditional --data data/ --output outputs/
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['traisformer', 'ensemble', 'traditional', 'all'],
        default='ensemble',
        help='Model(s) to run'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Perform cross-validation (for ensemble model)'
    )
    
    parser.add_argument(
        '--submission',
        action='store_true',
        help='Generate submission file'
    )
    
    parser.add_argument(
        '--config-traisformer',
        type=str,
        default='config/traisformer_config.yaml',
        help='TrAISformer configuration file'
    )
    
    parser.add_argument(
        '--config-ensemble',
        type=str,
        default='config/ensemble_config.yaml',
        help='Ensemble configuration file'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data).exists():
        logger.error(f"Data directory does not exist: {args.data}")
        sys.exit(1)
    
    # Update config paths if provided
    runner = CompetitionRunner(args)
    if args.config_traisformer:
        runner.configs['traisformer'] = args.config_traisformer
    if args.config_ensemble:
        runner.configs['ensemble'] = args.config_ensemble
    
    # Run competition
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("Competition interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Competition failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 