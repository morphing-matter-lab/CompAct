1. Contact:
Date: Apr. 28. 2025
written by Humphrey Yang (hanliny@andrew.cmu.edu)
This file is subjected to changes.

1. Ownership

    This code is uploaded and maintained by the Morphing Matter Lab. Please cite this work with this DOI:
    https://doi.org/10.1145/3706598.3714307
    This implementation is for academic and non-commerical uses only. Please 
    contact the author and CMU's Center for Technology Transfer and 
    Enterprise Creation if you are interested in using this software for
     commercial uses. See "LICENSE.pdf" for more information.

2. Dependencies:

    2a. System requirements
    The design tool only runs on Windows because it requires Grasshopper and add-ons, which currently only supports Windows.
    
    2b. For the design tool
    Rhinoceros 3D version 8 SR18 with grasshopper. Other versions may be 
    compatible with the software but are not tested.
    UI plus version 1.9.4 for grasshopper (https://www.food4rhino.com/en/app/ui)
    Python 3.12.2 (installed through Anaconda3)
        scipy 1.13.1
        numpy 1.26.4
        sympy 1.13.2
        cvxpy 1.5.3

3. Usage instructions:

    3a. parameters.py
    This file contains the parameters used by the design tool.

    3b. Starting to use the software
    Install the dependencies. While installing Python and/or Anaconda, make 
    sure to add the executable to PATH environments.

    3c. Using the design tool
    Run server.py in command line to initialize the server. The server handles all backend modules and must be initialized before opening the grasshopper UI.
    Run Rhinoceros 3D and grasshopper to lauch the script "UI.gh".

4. Known issues:

    The system is a prototype intentded to demonstrate the algorithms and interactivity.
    The authors will maintain the implementation as much as they are available. 
    If you find any issues that are not listed here while using the tool, 
    please contact the authors.

    4a. Numerical precision
    The Rhino/grasshopper front end and the python backend uses different 
    versions of Python (2 and 3, respectively), which have very different float 
    number implementations and may sometimes leading to unexpected and erroneous 
    results during linear alebraic computations. The linear solvers may also be 
    unable to fiund solutions due to small numeric deviations. This can 
    potentially be resolved by tuning the error thresholds in parameters.py.

    4b. UI malfunctions
    The design tool may sometimes stall and become irresponsive to inputs 
    despite the users still being able to click on the command buttons. To 
    avoid this issue, make sure that after completing an action, the command 
    prompt in Rhino is not asking for further input before clicking on another 
    command. If users find themselves in such situation, press ESC to cancel 
    the ongoing command and proceed as normal. Otherwise, rerunning the UI 
    script in grasshopper may also resolve this problem, though the user may 
    have to redo the design.

    4c. Speed
    The algorithms runs in real time and the authors have optimized the 
    computations as much as possible. Yet, the current implementation is still 
    bottlenecked by the communication overhead between the front- and backend 
    python instances. I.e., the majority of the waiting time comes from the 
    software interface overhead.

    4d. Axis alignment
    The algorithm should work regardless of axis alignment. However, axis-
    aligned designs are preffered as the values and calculations are numerically 
    more stable. If the designs are not axis-aligned, the design tool will 
    generate it's own coordinate systems that best suits the design, which may 
    not be axis aligned. In this case, when interpreting the sensor 
    responsiveness diagrams, the x, y, z axis will be based on the system-
    generated coordinate system, not the model world coordinates.