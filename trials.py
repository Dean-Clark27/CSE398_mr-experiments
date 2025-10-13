# Runs a command file multiple times. Can compare against other experiments.

import os
import shutil
import subprocess
import argparse
import pandas
import shlex
import time
import sys

#TODO: Add "no_neg_ex" to the list of files to delete. --clear doesn't work on directories yet
FILES_TO_DELETE = {
	"test_files": [
		"rKB_model.pth",
		"mr_train_examples.csv",
		"unification_embeddings.pt",
		"uni_mr_model.pt",
		"unification_nodes_traversed.csv",
		"unity_data.csv",
		"standard_data.csv",
		"triplets.csv"
	],
  "other": [
    "vocab.pkl",
		"random_facts.txt",
		"randomKB.txt",
		"test_queries.txt",
		"train_queries.txt",
		"train_anchors.csv",
		"train_negatives.csv",
		"train_positives.csv",
		"result_table",
		"rule_classifier.pth",
		"all_facts.txt",
		"trace_log.txt",
		"timings.txt"
	]
}

def clear_files(files: list):
	"""
	Cleans given files from a working directory
	"""
	for file_name in files:
		file_path = os.path.join("./", file_name)
		if os.path.exists(file_path):
			os.remove(file_path)

def save_results(source, destination):
	os.makedirs(destination, exist_ok=True)

	l = []
	for files in FILES_TO_DELETE:
		for file_name in FILES_TO_DELETE[files]:
			l.append(file_name)

			source_path = os.path.join(source, file_name)
			destination_path = os.path.join(destination, file_name)
			if not os.path.exists(source_path):
				continue

			shutil.move(source_path, destination_path)

	for filename in os.listdir(source):
		if filename.endswith('.png'):
			source_file = os.path.join(source, filename)
			destination_file = os.path.join(destination, filename)
			shutil.move(source_file, destination_file)
	
	for filename in os.listdir(source):
		if filename.endswith('.csv'):
			source_file = os.path.join(source, filename)
			destination_file = os.path.join(destination, filename)
			shutil.move(source_file, destination_file)


def copy_contents(source, destination):
	if not os.path.exists(os.path.join(os.getcwd(), source)):
		print(f"The folder '{source}' does not exist.")
		return

	os.makedirs(destination, exist_ok=True)

	for item in os.listdir(source):
		item_path = os.path.join(source, item)
		if not os.path.exists(item_path):
			continue

		destination_path = os.path.join(destination, item)

		if os.path.isdir(item_path):
			shutil.copytree(item_path, destination_path, dirs_exist_ok=True)
		else:
			shutil.copy2(item_path, destination_path)

def run_commands(command_file):
	commands = []

	with open(command_file, 'r') as file:
		lines = file.readlines()

	for cmd in lines:
		cmd = cmd.strip()
		commands.append(cmd)

	for command in commands:
		try:
			p = subprocess.run(
				shlex.split(command), capture_output=False, text=True, shell=False)
		except Exception as e:
			p.terminate()
			print(f"Command: |{command}| failed to execute.")
			print(e)
			print("Stopping trials...")
			sys.exit(1)

if __name__ == "__main__":
	aparser = argparse.ArgumentParser()
	aparser.add_argument("-c", "--clear", help="From two options: clears either 'test' results or 'all' results and models. Default during trial runs: all")
	aparser.add_argument("--commands_file", help="Path to commands that need to be executed. (E.g commands.txt)")
	aparser.add_argument("-n", "--number_of_executions", type=int, default=1, help="Number of times to execute commands_file. Default: 1")
	aparser.add_argument("--compare_trials", help="Folder path. Given a synthetic_results folder, runs a new trial against each trial inside found. Copies all results/models from the previous trial into the new trial.")
	aparser.add_argument("--trial_data", help="Folder path. Given a synthetic_results folder, calculates trial data and leaves csv in folder titled 'data.csv'")

	# TODO: Implement copy_experiment
	aparser.add_argument("--copy_experiment", help="Takes a trial folder and copies (kb, unification model, guidance model, test_queries, and vocab) to current working directory for execution with pathguidance_improved.")
	args = aparser.parse_args()

	if args.commands_file:
		trial_start_time = time.time()
		inst = time.strftime("%m.%d_%H.%M")
		synthetic_result_folder = f"./synthetic_results_{inst}/"
		os.makedirs(synthetic_result_folder, exist_ok=True)

		synthetic_results = []
		trial_count = args.number_of_executions

		if args.compare_trials:
			if not os.path.exists(os.path.join(os.getcwd(), args.compare_trials)):
				print(f"The synthetic data folder '{args.compare_trials}' does not exist.")
				sys.exit(1)

			for subfolder in os.listdir(args.compare_trials):
				if os.path.isdir(os.path.join(args.compare_trials, subfolder)):
					synthetic_results.append(subfolder)

			trial_count = len(synthetic_results)

		for i in range(trial_count):
			trial = i + 1
			trial_time = time.time()
			start_time = time.strftime("%m.%d_%H.%M")
			trial_folder = synthetic_result_folder + f"trial_{trial}_{start_time}"
			os.makedirs(trial_folder, exist_ok=True)

			if args.compare_trials:
				print("Copying previous trial data... (--compare_trials)")
				copy_contents(args.compare_trials + "/" + synthetic_results[i], "./")
				print("Done")
				time.sleep(5)
			
			print(f"Begin trial {trial} of {trial_count}")
			run_commands(args.commands_file)
			time.sleep(3) # Wait for files from above commands to finish being written
			save_results("./", trial_folder)
			if args.compare_trials:
				with open(trial_folder + "/against.txt", "w") as file:
					file.write(f"Trial {trial} was compared to: {synthetic_results[i]}. Trial seconds elapsed: {round(time.time() - trial_time, 2)}")

			time.sleep(3)			
			clear_files(FILES_TO_DELETE["test_files"])
			clear_files(FILES_TO_DELETE["other"])

			print(f"Trial {trial} completed, seconds elapsed: {round(time.time() - trial_time, 2)}\n\n")

		print(f"Trials Complete\nExecution time (in seconds): {round(time.time() - trial_start_time, 2)}")
	
	if args.trial_data:
		if os.path.isdir(args.trial_data):
			data_df = pandas.DataFrame({"Median nodes": [], "Mean nodes": []})

			for subfolder in os.listdir(args.trial_data):
				p = os.path.join(args.trial_data, subfolder)

				if os.path.isdir(p):
					#TODO: Change hard coded path to unification_nodes_traversed
					df = pandas.read_csv(os.path.join(p, "unification_nodes_traversed.csv"))
					num_nodes = len(df["Nodes Traversed"])
					mean = 0

					for nodes_traversed in df['Nodes Traversed']:
						mean += nodes_traversed
					mean /= num_nodes

					df = df.sort_values(by=['Nodes Traversed'])
					new_row = {"Median nodes": list(df['Nodes Traversed'])[int(num_nodes/2)], "Mean nodes": mean}
					data_df = pandas.concat([data_df, pandas.DataFrame([new_row])], ignore_index=True)
					print(f"Mean for '{subfolder}': {mean}")
			
			data_df.to_csv(args.trial_data + "/data.csv", index=False)
		else:
			print("Not a directory (--trial_data)")

	if args.clear:
		if isinstance(args.clear, str) and args.clear.lower() == "test":
			clear_files(FILES_TO_DELETE["test_files"])
		else:
			for x in FILES_TO_DELETE:
				clear_files(FILES_TO_DELETE[x])