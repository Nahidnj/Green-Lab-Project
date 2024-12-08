

#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
import GPUtil
import psutil
import subprocess
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv

# Sample prompts for different tasks
prompts = {
    "classification": [
        "Classify the sentiment of the following text: 'I love this product!'",
        "Is the following review positive or negative? 'The service was terrible.'"
    ],
    "sentiment_analysis": [
        "Analyze the sentiment of this tweet: 'Had a great day at the beach!'",
        "What is the sentiment of this statement: 'I am very disappointed with my purchase.'"
    ],

    "question_answering": [
        "What is the capital of France?",
        "Who wrote 'Pride and Prejudice'?"
    ]
}

# Function to get GPU metrics (utilization and VRAM usage)
def get_gpu_metrics():
    gpus = GPUtil.getGPUs()
    gpu_metrics = []
    for gpu in gpus:
        gpu_metrics.append({
            'gpu_id': gpu.id,
            'gpu_utilization': gpu.load * 100,
            'vram_usage': gpu.memoryUsed
        })
    return gpu_metrics

# Function to get GPU power
def get_gpu_power():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        power = float(result.stdout.strip())
        return power
    except Exception as e:
        print(f"Error getting GPU power: {e}")
        return None

# Function to get CPU metrics (utilization and frequency)
def get_cpu_metrics():
    cpu_utilization = psutil.cpu_percent(interval=1)
    cpu_power = psutil.cpu_freq().current
    return cpu_utilization, cpu_power

# Function to get CPU power using RAPL on Linux
def get_cpu_power():
    try:
        # This method uses RAPL to read CPU power consumption on Linux
        # Read the power data from the system file corresponding to the CPU package
        power_file = '/sys/class/powercap/intel-rapl:0/energy_uj' 

        with open(power_file, 'r') as f:
            energy_uj = int(f.read().strip())

        # RAPL returns energy in microjoules
        power_watt = energy_uj / 1_000_000  # Convert microjoules to joules
        return power_watt

    except Exception as e:
        print(f"Error getting CPU power: {e}")
        return None

# Function to run the model and return the response
def run_model(prompt, model_name, access_token):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).to(device)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"GPU memory insufficient for {model_name}. Falling back to CPU.")
        
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Main function to collect results and save them in a CSV
def main():
    access_token = "your_api_token"  # Replace with your actual token
    models = ["Qwen/Qwen2.5-7B", "google/gemma-2-2b-it", "mistralai/Mistral-7B-Instruct-v0.3"]

    with open('model_performance.csv', mode='w', newline='') as csvfile:
        fieldnames = ['Task', 'Prompt', 'Model', 'Response', 
                      'Duration (s)', 'GPU Utilization (%)', 
                      'GPU Power (W)', 'VRAM Usage (MB)', 
                      'CPU Utilization (%)', 'CPU Power (W)']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task, task_prompts in prompts.items():
            for prompt in task_prompts:
                for _ in range(30):  # Repeat each prompt 30 times
                    print(f"\nTesting {task} with prompt: '{prompt}'")

                    for model in models:
                        try:
                            # Collect metrics before running the model
                            start_time = time.time()
                            gpu_data_before = get_gpu_metrics()
                            gpu_power_before = get_gpu_power()
                            cpu_utilization_before, cpu_power_before = get_cpu_metrics()

                            # Get the CPU power using the new method
                            cpu_power = get_cpu_power()

                            response = run_model(prompt, model, access_token)

                            # Collect metrics after running the model
                            gpu_data_after = get_gpu_metrics()
                            gpu_power_after = get_gpu_power()
                            cpu_utilization_after, cpu_power_after = get_cpu_metrics()

                            # Calculate duration
                            duration = time.time() - start_time
                            
                            # Log metrics to CSV
                            writer.writerow({
                                'Task': task,
                                'Prompt': prompt,
                                'Model': model,
                                'Response': response,
                                'Duration (s)': duration,
                                'GPU Utilization (%)': gpu_data_after[0]['gpu_utilization'] if gpu_data_after else 'N/A',
                                'GPU Power (W)': gpu_power_after if gpu_power_after is not None else 'N/A',
                                'VRAM Usage (MB)': gpu_data_after[0]['vram_usage'] if gpu_data_after else 'N/A',
                                'CPU Utilization (%)': cpu_utilization_after,
                                'CPU Power (W)': cpu_power if cpu_power is not None else 'N/A'
                            })

                            print(f"\nModel: {model}")
                            print(f"Response: {response}")
                            print(f"Duration: {duration:.2f} seconds")
                            if gpu_data_after:
                                print(f"GPU Utilization: {gpu_data_after[0]['gpu_utilization']}%")
                                print(f"VRAM Usage: {gpu_data_after[0]['vram_usage']} MB")
                            if gpu_power_after is not None:
                                print(f"GPU Power: {gpu_power_after} W")
                            print(f"CPU Utilization: {cpu_utilization_after}%")
                            print(f"CPU Power: {cpu_power}")

                        except Exception as e:
                            print(f"Error processing model {model}: {e}")

# Run the main function
if __name__ == "__main__":
    main()
