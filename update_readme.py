import json
import re
from datetime import datetime

def update_readme_with_results():
    # Read the training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    # Get the best accuracy
    best_accuracy = max(entry['accuracy'] for entry in history)
    
    # Read the current README
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Update the Results section
    current_date = datetime.now().strftime("%Y-%m-%d")
    results_section = f"""
## Latest Results
- Test Accuracy: {best_accuracy:.2f}%
- Last Updated: {current_date}
- Training History:

Epoch 1: {history[0]['accuracy']:.2f}%
Epoch 5: {history[4]['accuracy']:.2f}%
Epoch 10: {history[9]['accuracy']:.2f}%
Final: {history[-1]['accuracy']:.2f}%

"""
    
    # Replace the old results section or append if it doesn't exist
    if '## Latest Results' in readme_content:
        pattern = r'## Latest Results.*?(?=##|$)'
        readme_content = re.sub(pattern, results_section, readme_content, flags=re.DOTALL)
    else:
        readme_content += '\n' + results_section
    
    # Write the updated README
    with open('README.md', 'w') as f:
        f.write(readme_content)

if __name__ == '__main__':
    update_readme_with_results() 