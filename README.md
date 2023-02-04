# Simple and efficient A★ python implementation

![](screenshot.gif)

The goal of this little project is to implement the A★ algorithm to find the shortest path in a efficient way:
- Each tile of the world have a value, the higher the value, the more expensive it is to take the way
- There are small monsters that walk around the map and also influence the weight of the tiles.
- The heuristic used for the algorithm is the Euclidean distance.
- For the performance the following data structures were used:
  - A heap to get the smallest element in log(n).
  - A dictionary to partition nodes by position, with random time access.

To start the project you simply install the dependencies with `pip install -r requirements.txt` and run the `python main.py` file. Each time a script is run, a map is selected from 100 possibilities.