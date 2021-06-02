The implementation of each type of method is seperated into two different folders named "splitting" and "kinetic_diffusion".
Those folders contain the implementation of the methods in the thesis. The script "main.py" contains the examples from the thesis
and shows how to use the implemented methods. main.py has a lot of options and for help please run main-py -h. Below are some examples
of how to run the main script.

To run the convergence and complexity tests from the final numerical experiment run the following command:
python main.py 0.1 0 1 --ml_test_APS --folder myfolder -sf --paths 180000 --type both

The above means that the Multilevel test are done for the APS method. The first number is the value of epsilon, the second is 
the value of a and the last on is b. --type specifies of both complexity and convergence test should run or only one of them. -sf means the the results should be saved and because of --folder, the results are saved
in "myfolder". The folder must exist in the working directory of main.py. Finally --paths 180000 specifies that results should be run with
180000 paths. 

To run it for other versions of the splitting approach optional 
arguments are available. For instance to run it for APSD add the argument -diff 