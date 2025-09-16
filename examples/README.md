1. Install instructions
for make: 
    From /reference-implementation/ run: "make"
    Files are created in different directories depending on use case. Tests will appear in /reference-implementation/out/. Exercises will appear in /reference-implementation/examples/ followed by their own /out/ directory connected to the exercise.
    For use of TBLIS (not needed for the exercise) clone from https://github.com/MatthewsResearchGroup/tblis and put it in /reference-implementation/tblis/
for cmake: (Unix commands)
    From /reference-implementation/ create subdirectory: "mkdir build"
    Enter directory: "cd build"
    Run CMake from directory: "cmake .."
    Run make from directory: "make -j"
    All files are created in the build directory
    For use of TBLIS (not needed for the exercise) add: "-DENABLE_TBLIS=1" after "cmake .."
With TBLIS a file called test++ will be compiled

2. Exercise contraction (try writing a tensor contraction with tapp)
The exercise is to follow the TODO list in examples/exercise_contraction/exercise_contraction.c, use examples/driver.c as help.
With make: when completed you can run "make" or "make exercise_contraction". This will create an executable file: examples/exercise_contraction/out/exercise_contraction
With cmake: the install commands for cmake. This will create an executable exercise_contraction in the build directory
If you are not able to solve it, you can find answers in examples/exercise_contraction/answers/exercise_contraction_answers.c, but please don't be afraid to ask us first.

3. Exercise tucker (try integrating tapp into a practical example)
Use a virtual environment:
    python -m venv .venv
    source .venv/bin/activate
Install the Python dependencies: numpy, matplotlib, tensorly(, PyQt6 if needed)
    python -m pip install numpy matplotlib tensorly
Run with python examples/exercise_tucker/numpy_tucker/exercise_tucker_numpy.py for a working numpy version
The exercise is to complete code making examples/exercise_tucker/tapp_tucker/exercise_tucker_tapp.py reproduce the functionality with tapp
Follow the TODO list in examples/exercise_tucker/tapp_tucker/exercise_tucker.c followed by the TODO list in examples/exercise_tucker/tapp_tucker/exercise_tucker_tapp.py
With make: when completed you can run "make" or "make exercise_tucker". This will create a shared object file: examples/exercise_tucker/tapp_tucker/lib/exercise_tucker.so
With cmake: the install commands for cmake. This will create a shared object exercise_tucker.so in the build directory
Run with python examples/exercise_tucker/tapp_tucker/exercise_tucker_tapp.py
If you are not able to solve it, you can find answers in examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_answers.c and examples/exercise_tucker/tapp_tucker/answers/exercise_tucker_tapp_answers.py, but please don't be afraid to ask us first.
OBS: If you use cmake copy example_img.png and exercise_tucker_tapp.py from examples/exercise_tucker/tapp_tucker/ to the build directory and remove "/lib" from line 18 and line 20.