# Gait Pattern Characterization Tool
Tool for gait pattern characterization analysis based on ROM (Range Of Motion) signals developed for my bachelor's thesis on Biomedical Engineering at San Pablo-CEU University on 2023. 

This work aims to develop a tool to identify and report gait parameters based on the range of hip joint motion obtained from robotic platforms. The tool relies on flexion and extension data obtained from the SWalker robotic platform, designed for rehabilitating patients with hip fractures, which are common in the elderly. The tool aims to provide useful information, beyond the range of motion, to physiotherapists responsible for directing rehabilitation sessions, helping them designing more specific sessions and monitor the patient's physical progress.
The platform SWalker measures the range of hip movement using potentiometers placed on a rigid structure at pelvis level. The tool designed on this project calculates the gait parameters: cadence, step length, stride length, step time, single support time, double support time, support time, stride time, walking speed, stride speed, and swing time.

Unfortunately, the step width cannot be measured as the data is only obtained in the sagital plane.
For more information about the project, read TFG_AlfonsoGordon

Packages needed:
 1. glob
 2. pandas
 3. os
 4. numpy
 5. scipy.signal
 6. matplotlib.pyplot
