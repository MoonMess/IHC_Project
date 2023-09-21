# Basic usage

## Installing requirements

First, create a new environment and acivated it :
```bash
python3 -m venv env
source env/bin/activate (Unix/macOS)
.\env\Scripts\activate (windows)
```
Then, intalled the required librairies : 
 
```bash
pip install -r requirements.txt
```

## Launching the project

```bash
python main.py [-s|--slide_folder] 
 ```
<br />
MANDATORY: <br />
- [-s|--slide_folder] Path to the folder cointaing the slides. Could be avoid by changing the DEFAULT_INPUT_SLIDE_FOLDER in parameters.py.
<br />


The outputs are the following : 
- saved plot in plot/Classification/"slide_name"/
<br /><br />
We are trying to classify the tumoral cells, between 0,1,2,3. The plots show the patch, the classifcation and some cells of each class.