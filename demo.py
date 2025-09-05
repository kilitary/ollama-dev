#!/usr/bin/env python3
"""
Ollama Development Toolkit - Example Usage

This script demonstrates basic usage of the toolkit's main features:
- Model management and configuration
- Text analysis with structured logging
- Language feature categorization
- Response processing concepts

Run this script to verify your installation and see the toolkit in action.
"""

import sys
import os
import re
import time
import random
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import availability flags
MODULES_AVAILABLE = {
    'analyze': False,
    'config': False, 
    'langfeatures': False,
    'sim': False,
    'ollama': False
}

# Try importing main modules
try:
    from analyze import slog, update_model
    MODULES_AVAILABLE['analyze'] = True
except ImportError:
    pass

try:
    from config import selected_model, temperature, src_options
    MODULES_AVAILABLE['config'] = True
except ImportError:
    # Fallback values
    selected_model = 'mistral-nemo:latest'
    temperature = 0.55
    src_options = {
        'num_ctx': 8192, 'top_k': 20, 'top_p': 0.9, 
        'repeat_penalty': 1.2, 'num_thread': 11, 'use_mmap': True
    }

try:
    from langfeatures import features
    MODULES_AVAILABLE['langfeatures'] = True
except ImportError:
    # Fallback features
    features = {
        1: ['sort', 'encode', 'handle', 'process', 'analyze'],
        2: ['system', 'device', 'component', 'engine', 'model'], 
        3: ['fast', 'clean', 'optimized', 'available', 'working']
    }

try:
    import ollama
    MODULES_AVAILABLE['ollama'] = True
except ImportError:
    pass


def log_func(msg, style=None, justify=None, end="\n"):
    """Logging function that works with or without Rich."""
    if MODULES_AVAILABLE['analyze']:
        slog(msg, style=style, justify=justify, end=end)
    else:
        # Strip Rich markup for plain text output
        clean_msg = re.sub(r'\[/?[^\]]*\]', '', str(msg))
        print(clean_msg, end=end)


def demonstrate_logging():
    """Demonstrate the structured logging system."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Structured Logging System")
    print("="*60)
    
    if not MODULES_AVAILABLE['analyze']:
        log_func("Note: Using fallback logging (Rich not available)", style="yellow")
    
    log_func("Welcome to Ollama Development Toolkit!", style="bold green")
    log_func("This is a [red]colored[/red] message with [blue]Rich markup[/blue]")
    log_func("Left aligned text", justify="left", style="cyan")
    log_func("Center aligned text", justify="center", style="yellow")  
    log_func("Right aligned text", justify="right", style="magenta")
    log_func("Processing", end="", style="blue")
    time.sleep(0.5)
    log_func("...", end="", style="blue") 
    time.sleep(0.5)
    log_func(" Complete!", style="green")


def demonstrate_model_management():
    """Demonstrate model management functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Model Management")
    print("="*60)
    
    log_func(f"Current selected model: [cyan]{selected_model}[/cyan]")
    log_func(f"Temperature setting: [yellow]{temperature}[/yellow]")
    log_func(f"Context window size: [blue]{src_options['num_ctx']}[/blue] tokens")
    
    # Check if Ollama is available
    if MODULES_AVAILABLE['ollama']:
        try:
            client = ollama.Client()
            log_func("✓ Ollama client connection successful", style="green")
            
            # Try to list available models
            models = client.list()
            if models and 'models' in models:
                log_func(f"Available models: {len(models['models'])}")
                for model in models['models'][:3]:  # Show first 3 models
                    log_func(f"  - {model['name']}", style="dim")
            else:
                log_func("No models found. Try: ollama pull mistral-nemo:latest", style="yellow")
                
        except Exception as e:
            log_func(f"✗ Ollama connection failed: {e}", style="red")
            log_func("Make sure Ollama is running: ollama serve", style="yellow")
    else:
        log_func("Ollama module not available", style="yellow")
        log_func("Install with: pip install ollama", style="dim")


def demonstrate_language_features():
    """Demonstrate language feature categorization."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Language Feature Categories")
    print("="*60)
    
    log_func("Language features are organized into categories:")
    
    categories = {
        1: "Action verbs", 
        2: "Nouns/objects",
        3: "Descriptive adjectives"
    }
    
    for cat_id, description in categories.items():
        if cat_id in features:
            sample = random.sample(features[cat_id], min(3, len(features[cat_id])))
            log_func(f"Category {cat_id} - {description}: [cyan]{', '.join(map(str, sample))}[/cyan]")


def demonstrate_configuration():
    """Demonstrate configuration options."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Configuration System")  
    print("="*60)
    
    log_func("Key configuration parameters:")
    
    config_highlights = {
        "Model": selected_model,
        "Temperature": temperature,
        "Context Size": src_options["num_ctx"],
        "Top-K Sampling": src_options["top_k"],
        "Top-P (Nucleus)": src_options["top_p"],
        "Repetition Penalty": src_options["repeat_penalty"],
        "CPU Threads": src_options["num_thread"],
        "Memory Mapping": src_options["use_mmap"]
    }
    
    for param, value in config_highlights.items():
        log_func(f"  {param}: [cyan]{value}[/cyan]")


def demonstrate_response_filtering():
    """Demonstrate response filtering concepts."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Response Processing & Filtering")
    print("="*60)
    
    # Simulate response processing
    sample_responses = [
        "This is a normal response that would pass through.",
        "I would like some milk and cookies.",  # Contains stop sign
        "As a responsible AI, I cannot help with that.",  # Contains keyword
        "Here's a detailed technical analysis of the system."
    ]
    
    stop_signs = ['milk', 'egg', 'food', 'tea', 'cake']
    keywords = ['responsible ai', 'violates ethical', 'potentially harmful']
    
    log_func("Processing sample responses:")
    
    for i, response in enumerate(sample_responses, 1):
        log_func(f"\nResponse {i}: [dim]{response[:50]}...[/dim]")
        
        # Check for stop signs
        triggered_stop = any(sign in response.lower() for sign in stop_signs)
        if triggered_stop:
            log_func("  ⚠️  Stop sign detected - would trigger reset", style="yellow")
            continue
            
        # Check for keywords
        triggered_keyword = any(keyword in response.lower() for keyword in keywords)
        if triggered_keyword:
            log_func("  🔍 Special keyword detected - would apply special handling", style="blue")
            continue
            
        log_func("  ✅ Response passes filtering - would display normally", style="green")


def show_module_status():
    """Show which modules are available."""
    print("\n" + "="*60)
    print("MODULE AVAILABILITY STATUS")
    print("="*60)
    
    for module, available in MODULES_AVAILABLE.items():
        status = "✓ Available" if available else "✗ Not Available"
        color = "green" if available else "red"
        log_func(f"{module:12}: {status}", style=color)
        
    if not any(MODULES_AVAILABLE.values()):
        log_func("\nNote: Running in demonstration mode with fallback implementations", style="yellow")


def main():
    """Run all demonstrations."""
    print("OLLAMA DEVELOPMENT TOOLKIT - DEMONSTRATION")
    print("This script showcases the main features of the toolkit.")
    
    try:
        show_module_status()
        demonstrate_logging()
        demonstrate_model_management()
        demonstrate_language_features()
        demonstrate_configuration()
        demonstrate_response_filtering()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        log_func("All demonstrations completed successfully! 🎉", style="bold green")
        log_func("For more information, see the documentation in the [cyan]docs/[/cyan] directory")
        log_func("To start using the toolkit:", style="yellow")
        log_func("  • Run [cyan]python analyze.py[/cyan] for analysis")
        log_func("  • Run [cyan]python sim.py[/cyan] for simulations")
        log_func("  • Check [cyan]docs/[/cyan] for detailed documentation")
        
    except KeyboardInterrupt:
        log_func("\nDemonstration interrupted by user", style="yellow")
    except Exception as e:
        log_func(f"\nError during demonstration: {e}", style="red")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)