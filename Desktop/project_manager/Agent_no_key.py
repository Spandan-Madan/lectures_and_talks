from flask import Flask, render_template, flash, request, redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pickle
from pathlib import Path
import json
import os
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

cwd = os.getcwd()
PATH_DATA = '%s/Agent/projects.pckl'%cwd

class ReusableForm(Form):
	name = TextField('Name:', validators = [validators.required()])

@app.route("/reset", methods = ['GET','POST'])
def reset_todo():
	form = ReusableForm(request.form)
	print(form.errors)
	# if request.method == 'POST':
		# project_name = request.form['Project_Name']
		# task_name = request.form['Task_Name']
	delete_TODOS()
	return redirect('http://localhost:5000/')

	show_existing()

@app.route("/", methods = ['GET','POST'])
def main_function():
	form = ReusableForm(request.form)

	print(form.errors)
	if request.method == 'POST':
		project_name = request.form['Project_Name']
		task_name = request.form['Task_Name']
		delete_task = request.form['Delete_Task']

		if len(project_name) == 0 and len(task_name) == 0:
			show_existing()
		else:
			Add_new(project_name,task_name,delete_task)
	return render_template('new_project.html', form=form)


def Add_new(project_name,task_name,delete_task):
		
		cwd = os.getcwd()
		if Path(PATH_DATA).is_file():
			data_dict = read_existing()

			if project_name in data_dict.keys():
				if len(task_name) > 0:
					data_dict[project_name].append(task_name)
					flash(task_name + ' added to ' + project_name)
					save_new(data_dict)
					show_existing()

				elif len(delete_task) > 0:
					data_dict[project_name].pop(int(delete_task)-1)
					save_new(data_dict)
					show_existing()

				else:
					flash('Tasks in project are - ')
					tasks_in_project = str(data_dict[project_name])
					flash(tasks_in_project)
			else:
				if len(task_name) > 0:
					data_dict[project_name] = [task_name]
					flash(project_name + ' added and ' + task_name + ' added to it.')
					save_new(data_dict)
					show_existing()
				else:
					data_dict[project_name] = []
					flash(project_name + ' created')
					save_new(data_dict)
					show_existing()


		else:
			data_dict = {}

			if len(task_name) > 0:
				data_dict[project_name] = [task_name]
				flash('dataset created and ' + project_name + ' added and ' + task_name + ' added to it.')
				save_new(data_dict)
				show_existing()
			else:
				data_dict[project_name] = []
				flash('dataset created and ' + project_name + ' created')
				save_new(data_dict)
				show_existing()

def read_existing():
	f = open(PATH_DATA, 'rb')
	data_dict = pickle.load(f)
	f.close()
	return data_dict

def save_new(data_dict):
	f = open(PATH_DATA, 'wb')
	pickle.dump(data_dict,f)
	f.close()

def show_existing():
	if Path(PATH_DATA).is_file():
		f = open(PATH_DATA, 'rb')
		data_dict = pickle.load(f)
		f.close()
		# str_json = json.dumps(data_dict)
		for key in data_dict.keys():
			flash('Project - ' + key)
			counter = 0
			for task in data_dict[key]:
				counter += 1
				flash(str(counter) + '. ' + task)
			flash('____________________________________________________________')
	else:
		flash('No data exists yet, please add some!')

def delete_TODOS():
	f = open(PATH_DATA, 'rb')
	data_dict = pickle.load(f)
	f.close()
	data_dict.pop('TODOS',None)
	save_new(data_dict)
	show_existing()

if __name__ == "__main__":
	app.run()
