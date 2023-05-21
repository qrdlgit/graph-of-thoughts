import subprocess
import os
import signal
import sys
import oai
import time
import traceback

def signal_handler(sig, frame):
    print('You interrupted the process!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
insights = []

def get_best_model(base_source_filename, prev_best):
    global insights
    # Read the contents of the base source file
    with open(base_source_filename, 'r') as f:
        base_source_contents = f.read()

    best_score = -1
    best_script_filename = None
    # Loop over 3 iterations
    for i in range(3):
        flag = True
        time_wait = 2
        while flag:
            try:
                insight = ""
                # Generate new filename
                new_script_filename = f'{base_source_filename[0:-3]}_n{i}.py'
                insight_text = "\n".join(insights)
                # Generate prompt and get response
                prompt = f"Here are some insights you previously made:\n"+insight_text+f"\n\nPlease make a significant attempt to improve the r2_score metric of the following code. Utilize the insights previously made, but do more than just repeat them and making minor hyperparameter adjustments. The code should output two lines.  One line should be 'r2_score: <new_r2_score> and another single line saying 'Insight: <detailed description of change made over previous code, make sure you say what was in current source({base_source_filename}) versus what will now be in source you're creating({new_script_filename}) which is the new code> causes the r2_score to go from {prev_best} to <new_r2_score>'. Make sure you include both filenames, the current and the new source in the output.  Make sure you always use at least 3 significant decimal digits. Outside the new code, do not provide an explanation, just the code and no additional text.\n\n"
                print(prompt)
                prompt = prompt + base_source_contents
                response = oai.get_response(prompt)
                
                # Remove boundary characters if present
                response = response.replace('```', '')
                
                # Write response to new file
                with open(new_script_filename, 'w') as f:
                    f.write(response)

                print(f"executing {new_script_filename},", end=" ")
                # Execute the new script and get output
                result = subprocess.run(['python3', new_script_filename], capture_output=True, text=True)
                
                # Extract r2_score from the output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'r2_score:' in line:
                        score = float(line.split(':')[-1].strip())
                    if 'Insight' in line:
                        insight = line
                   
                        # Update best score and script if this score is better
                    if score > best_score:
                        best_score = score
                        best_script_filename = new_script_filename

                print(f"r2_score: {score} {insight}")
                flag = False
            except Exception as e:
                time_wait = time_wait * 2
                print(f"error with {new_script_filename}, waiting {time_wait} and retrying")
                traceback.print_exc()
                time.sleep(time_wait)
            if insight != '':
                insights = insights + [insight]

    # Recursive call
    if best_script_filename:
         get_best_model(best_script_filename, best_score)

# Run the recursive function
base_source_filename = 'base.py'
get_best_model(base_source_filename, 0.575)
