
SignalProcess project
=====================

The program is intended for initial processing of experimental data (digital signals of detectors).

Key features:
-------------

* reading data from CSV (DAT, TXT) files in different formats, as well as WFM files
* multiplying by the specified multiplier and subtracting the specified delay independently for each signal
* alignment of the time origin on the front edge of the selected signal (binding)
* automatic subtraction of a constant component from the amplitude of selected signals
* search for signals' peaks with flexible adjustment of search parameters
* automatic grouping of peaks (from different signals), coinciding (with the specified tolerance) in time
* plotting of signals' graphs (preview)
* construction of graphs with several signals and their peaks (with one axis of time, and several axes of amplitudes)
