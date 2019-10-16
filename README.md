## Example 6 - Tensorflow < 1.5 + MNIST

**Definition:** Vanilla Universe job to train a simple neural network using TensorFlow 

**Objective:** User learns to submit a script which calls a python executable on project spaces to train a deep network using TensorFlow. 


- On a managed machine ssh into the condor master:
```shell
ssh condor
```
- Change to your condor-examples dir and submit the condor job:
```shell
cd ~/condor-examples/Vanilla/Example06/
condor_submit example06.submit_file -interac
```
- This will train a basic Tensorflow network on the MNIST data, using the an anaconda python environment stored on a project space.