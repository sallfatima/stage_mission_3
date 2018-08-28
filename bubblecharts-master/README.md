# traveller-bubble
Bubble Graphs with Traveller (sub) sector data

This program uses an input file that requires a Law Level, Acceptance Level (in the Cx data), Tech Level, and Starport.

There is a current example of the file to follow as a model called bubblesec.txt.

When copying from Travellermap.com strip out everything before the header line.

You must use a header line with the Travellermap.com headers.  
After reading the header the program identifies which column number to find the key stats and starts looking for them on line 3 (ignoring the second line as that is a series of dashes).

It was built to copy and paste the input data into a file called bubblesec.txt.  
