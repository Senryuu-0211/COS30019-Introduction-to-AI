import sys
from typing import List, Tuple, Set
from collections import deque
from heapq import heappush, heappop
import itertools

class Node:
    def __init__(self, state: Tuple[int, int], parent=None, action=None, path_cost=0):
        self.state = state  # (x, y) coordinates
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

class RobotNavigation:
    def __init__(self, filename: str):
        self.grid_size = None  # (rows, cols)
        self.initial_state = None  # (x, y)
        self.goal_states = []  # list of (x, y)
        self.walls = []  # list of (x, y, w, h)
        self.visited_nodes = set() # track visited nodes
        self.load_problem(filename)
        
    def load_problem(self, filename: str):
        """Load problem from file"""
        try:
            with open(filename, 'r') as f:
                # Parse grid size [N,M]
                line = f.readline().strip()
                if not (line.startswith('[') and line.endswith(']')):
                    raise ValueError("Invalid grid size format. Expected [N,M]")
                n, m = map(int, line.strip('[]').split(','))
                self.grid_size = (n, m)
                
                # Parse initial state (x,y)
                line = f.readline().strip()
                if not (line.startswith('(') and line.endswith(')')):
                    raise ValueError("Invalid initial state format. Expected (x,y)")
                x, y = map(int, line.strip('()').split(','))
                self.initial_state = (x, y)
                
                # Parse goal states (x,y) | (x,y) | ...
                line = f.readline().strip()
                goal_states = line.split('|')
                self.goal_states = []
                for goal in goal_states:
                    goal = goal.strip()
                    if not (goal.startswith('(') and goal.endswith(')')):
                        raise ValueError("Invalid goal state format. Expected (x,y)")
                    x, y = map(int, goal.strip('()').split(','))
                    self.goal_states.append((x, y))
                
                # Parse walls (x,y,w,h)
                self.walls = []
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        if not (line.startswith('(') and line.endswith(')')):
                            raise ValueError("Invalid wall format. Expected (x,y,w,h)")
                        x, y, w, h = map(int, line.strip('()').split(','))
                        self.walls.append((x, y, w, h))
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Problem file '{filename}' not found")
        except ValueError as e:
            raise ValueError(f"Error parsing problem file: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading problem file: {e}")

    def get_valid_moves(self, state: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """Returns list of valid (action, new_state) pairs"""
        x, y = state
        moves = []
        # Order: UP, LEFT, DOWN, RIGHT
        directions = [
            ('up', (x, y-1)),
            ('left', (x-1, y)),
            ('down', (x, y+1)),
            ('right', (x+1, y))
        ]
        
        for action, new_state in directions:
            if self.is_valid_state(new_state):
                moves.append((action, new_state))
        return moves

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is valid (within grid and not in wall)"""
        x, y = state
        # Check grid boundaries
        if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
            return False
            
        # Check walls
        for wall_x, wall_y, width, height in self.walls:
            if (wall_x <= x < wall_x + width) and (wall_y <= y < wall_y + height):
                return False
        return True

    def is_goal(self, state: Tuple[int, int]) -> bool:
        """Check if state is a goal state"""
        return state in self.goal_states

    def get_path(self, node: Node) -> List[str]:
        """Reconstruct path from start to goal"""
        path = []
        current = node
        while current.parent:
            path.append(current.action)
            current = current.parent
        return path[::-1]
    
    def reset_visited_nodes(self):
        """Reset the set of visited nodes"""
        self.visited_nodes = set()

    def add_visited_node(self, state: Tuple[int, int]):
        """Add a node to the set of visited nodes"""
        self.visited_nodes.add(state)

    def get_visited_nodes(self) -> Set[Tuple[int, int]]:
        """Return the set of visited nodes"""
        return self.visited_nodes

    def dfs(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """Depth-first search implementation"""
        self.reset_visited_nodes()
        start_node = Node(self.initial_state)
        frontier = [start_node]
        explored = set()
        nodes_created = 1

        while frontier:
            node = frontier.pop()
            self.add_visited_node(node.state)
            
            if self.is_goal(node.state):
                return node.state, nodes_created, self.get_path(node)
                
            explored.add(node.state)
            
            for action, next_state in self.get_valid_moves(node.state):
                if next_state not in explored:
                    child = Node(next_state, node, action, node.path_cost + 1)
                    frontier.append(child)
                    nodes_created += 1

        return None, nodes_created, []

    def bfs(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """Breadth-first search implementation"""
        self.reset_visited_nodes()
        start_node = Node(self.initial_state)
        frontier = deque([start_node])
        explored = set()
        nodes_created = 1

        while frontier:
            node = frontier.popleft()
            self.add_visited_node(node.state)
            
            if self.is_goal(node.state):
                return node.state, nodes_created, self.get_path(node)
                
            explored.add(node.state)
            
            for action, next_state in self.get_valid_moves(node.state):
                if next_state not in explored and next_state not in [n.state for n in frontier]:
                    child = Node(next_state, node, action, node.path_cost + 1)
                    frontier.append(child)
                    nodes_created += 1

        return None, nodes_created, []

    def manhattan_distance(self, state: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic"""
        x, y = state
        return min(abs(x - gx) + abs(y - gy) for gx, gy in self.goal_states)

    def astar(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """A* search implementation"""
        self.reset_visited_nodes()
        start_node = Node(self.initial_state)
        frontier = [(self.manhattan_distance(start_node.state), start_node)]
        explored = set()
        nodes_created = 1

        while frontier:
            _, node = heappop(frontier)
            self.add_visited_node(node.state)
            
            if self.is_goal(node.state):
                return node.state, nodes_created, self.get_path(node)
                
            explored.add(node.state)
            
            for action, next_state in self.get_valid_moves(node.state):
                if next_state not in explored:
                    child = Node(next_state, node, action, node.path_cost + 1)
                    f = child.path_cost + self.manhattan_distance(child.state)
                    heappush(frontier, (f, child))
                    nodes_created += 1

        return None, nodes_created, []
    
    def gbfs(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """Greedy Best-First Search implementation"""
        self.reset_visited_nodes()
        start_node = Node(self.initial_state)
        frontier = [(self.manhattan_distance(start_node.state), start_node)]
        explored = set()
        nodes_created = 1

        while frontier:
            _, node = heappop(frontier)
            self.add_visited_node(node.state)
            
            if self.is_goal(node.state):
                return node.state, nodes_created, self.get_path(node)
                
            explored.add(node.state)
            
            for action, next_state in self.get_valid_moves(node.state):
                if next_state not in explored:
                    child = Node(next_state, node, action, node.path_cost + 1)
                    heappush(frontier, (self.manhattan_distance(child.state), child))
                    nodes_created += 1

        return None, nodes_created, []

    def iddfs(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """Iterative Deepening Depth-First Search implementation"""
        self.reset_visited_nodes()
        depth_limit = 0
        nodes_created = 0

        while True:
            result, new_nodes, path = self.depth_limited_search(depth_limit)
            nodes_created += new_nodes

            if result is not None:
                return result, nodes_created, path
            
            depth_limit += 1

    def depth_limited_search(self, depth_limit: int) -> Tuple[Tuple[int, int], int, List[str]]:
        """Depth-Limited Search helper function for IDDFS"""
        start_node = Node(self.initial_state)
        frontier = [start_node]
        explored = set()
        nodes_created = 1

        while frontier:
            node = frontier.pop()
            self.add_visited_node(node.state)
            
            if self.is_goal(node.state):
                return node.state, nodes_created, self.get_path(node)
            
            if node.path_cost < depth_limit:
                explored.add(node.state)
                
                for action, next_state in self.get_valid_moves(node.state):
                    if next_state not in explored:
                        child = Node(next_state, node, action, node.path_cost + 1)
                        frontier.append(child)
                        nodes_created += 1

        return None, nodes_created, []

    def astar_all_goals(self) -> Tuple[List[Tuple[int, int]], int, List[str]]:
        """A* search implementation to visit all goals"""
        self.reset_visited_nodes()
        start = self.initial_state
        unvisited = set(self.goal_states)
        path = [start]
        total_path = []
        nodes_created = 0

        while unvisited:
            nearest = min(unvisited, key=lambda x: self.manhattan_distance_between(start, x))
            result, new_nodes, segment = self.astar_between(start, nearest)
            nodes_created += new_nodes
            if result is None:
                return None, nodes_created, []
            path.extend(result[1:])  # Exclude the start point to avoid duplication
            total_path.extend(segment)
            unvisited.remove(nearest)
            start = nearest

        return path, nodes_created, total_path

    def manhattan_distance_between(self, state1: Tuple[int, int], state2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two states"""
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

    def astar_between(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], int, List[str]]:
        """A* search implementation between two points"""
        start_node = Node(start)
        frontier = [(0, start_node)]
        explored = set()
        nodes_created = 1

        while frontier:
            _, node = heappop(frontier)
            
            if node.state == goal:
                path = []
                actions = []
                while node:
                    path.append(node.state)
                    if node.action:
                        actions.append(node.action)
                    node = node.parent
                return path[::-1], nodes_created, actions[::-1]
                
            explored.add(node.state)
            
            for action, next_state in self.get_valid_moves(node.state):
                if next_state not in explored:
                    child = Node(next_state, node, action, node.path_cost + 1)
                    f = child.path_cost + self.manhattan_distance_between(child.state, goal)
                    heappush(frontier, (f, child))
                    nodes_created += 1

        return None, nodes_created, []

    def ida_star(self) -> Tuple[Tuple[int, int], int, List[str]]:
        """Iterative Deepening A* (IDA*) search implementation"""
        self.reset_visited_nodes()
        start_node = Node(self.initial_state)
        threshold = self.manhattan_distance(start_node.state)
        nodes_created = 0

        while True:
            result, new_nodes, path, new_threshold = self.ida_star_search(start_node, threshold)
            nodes_created += new_nodes

            if result is not None:
                return result, nodes_created, path
            
            if new_threshold == float('inf'):
                return None, nodes_created, []
            
            threshold = new_threshold

    def ida_star_search(self, node: Node, threshold: int) -> Tuple[Tuple[int, int], int, List[str], int]:
        """IDA* search helper function"""
        f = node.path_cost + self.manhattan_distance(node.state)
        if f > threshold:
            return None, 0, [], f

        self.add_visited_node(node.state)
        
        if self.is_goal(node.state):
            return node.state, 1, self.get_path(node), threshold

        min_threshold = float('inf')
        nodes_created = 1

        for action, next_state in self.get_valid_moves(node.state):
            child = Node(next_state, node, action, node.path_cost + 1)
            result, new_nodes, path, new_threshold = self.ida_star_search(child, threshold)
            nodes_created += new_nodes

            if result is not None:
                return result, nodes_created, path, new_threshold

            min_threshold = min(min_threshold, new_threshold)

        return None, nodes_created, [], min_threshold

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    
    # Create problem instance
    problem = RobotNavigation(filename)
    
    # Run specified search method
    search_methods = {
        'DFS': problem.dfs,
        'BFS': problem.bfs,
        'Astar': problem.astar,
        'GBFS': problem.gbfs,
        'IDDFS': problem.iddfs,
        'ALL': problem.astar_all_goals,
        'CUS2': problem.ida_star,
    }
    
    if method not in search_methods:
        print(f"Error: Unknown search method '{method}'")
        sys.exit(1)
    
    # Run search
    result = search_methods[method]()
    
    # Output results
    if result[0]:
        print(f"{filename} {method}")
        if method == 'ALL':
            path, nodes, actions = result
            print(f"<Path{path}> {nodes}")
            print(' -> '.join(actions))
        else:
            goal, nodes, path = result
            print(f"<Node{goal}> {nodes}")
            print(' -> '.join(path))
    else:
        print(f"{filename} {method}")
        print(f"No goal is reachable; {result[1]}")

if __name__ == "__main__":
    main()
