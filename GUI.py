import tkinter as tk
from customtkinter import *
from tkinter import filedialog, messagebox
import time
from typing import List, Tuple, Union
from search import RobotNavigation  # Assuming this is the file containing our RobotNavigation class

class RobotNavigationGUI(CTk):
    def __init__(self):
        super().__init__()
        self.title("Robot Navigation Simulation")
        set_appearance_mode("System")  # Interface mode based on system
        set_default_color_theme("blue")  # Default color theme

        self.problem = None
        self.cell_size = 40
        self.delay = 100  # Milliseconds between updates
        self.selected_goal = None
        self.animation_speed = tk.DoubleVar(value=1.0)

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = CTkFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control frame
        self.control_frame = CTkFrame(main_frame)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Load problem button
        CTkButton(self.control_frame, text="Load Problem", command=self.load_problem).pack(side=tk.LEFT, padx=5)

        # Algorithm selection
        CTkLabel(self.control_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algorithm = StringVar()
        algorithms = ['DFS', 'BFS', 'Astar', 'GBFS', 'IDDFS', 'IDA_STAR']
        CTkOptionMenu(self.control_frame, variable=self.algorithm, values=algorithms).pack(side=tk.LEFT, padx=5)
        self.algorithm.set(algorithms[0])

        # Run button for single goal
        CTkButton(self.control_frame, text="Run Single Goal", command=self.run_single_goal_algorithm).pack(side=tk.LEFT, padx=5)

        # Run button for multi goal
        CTkButton(self.control_frame, text="Run Multi Goal", command=self.run_multi_goal_algorithm).pack(side=tk.LEFT, padx=5)

        # Animation speed slider
        CTkLabel(self.control_frame, text="Animation Speed:").pack(side=tk.LEFT, padx=5)
        CTkSlider(self.control_frame, from_=0.1, to=2.0, variable=self.animation_speed).pack(side=tk.LEFT, padx=5)

        # Canvas for grid
        self.canvas = CTkCanvas(main_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Status bar
        self.status_bar = CTkLabel(main_frame, text="Ready", anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    def load_problem(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filename:
            try:
                self.problem = RobotNavigation(filename)
                self.draw_grid()
                self.status_bar.configure(text=f"Problem loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Unable to load problem: {str(e)}")

    def draw_grid(self):
        self.canvas.delete("all")  # Clear canvas
        if not self.problem:
            return

        width = self.problem.grid_size[1] * self.cell_size
        height = self.problem.grid_size[0] * self.cell_size
        self.canvas.config(width=width, height=height)

        # Draw grid lines
        for i in range(self.problem.grid_size[0] + 1):
            self.canvas.create_line(0, i * self.cell_size, width, i * self.cell_size, fill="lightgray")
        for j in range(self.problem.grid_size[1] + 1):
            self.canvas.create_line(j * self.cell_size, 0, j * self.cell_size, height, fill="lightgray")

        # Draw walls
        for wall in self.problem.walls:
            x, y, w, h = wall
            self.canvas.create_rectangle(
                x * self.cell_size, y * self.cell_size,
                (x + w) * self.cell_size, (y + h) * self.cell_size,
                fill="gray", outline="black"
            )

        # Draw initial state
        x, y = self.problem.initial_state
        self.canvas.create_oval(
            x * self.cell_size + 5, y * self.cell_size + 5,
            (x + 1) * self.cell_size - 5, (y + 1) * self.cell_size - 5,
            fill="red", outline="black", tags="robot"
        )

        # Draw goal states
        for i, goal in enumerate(self.problem.goal_states):
            x, y = goal
            self.canvas.create_rectangle(
                x * self.cell_size + 5, y * self.cell_size + 5,
                (x + 1) * self.cell_size - 5, (y + 1) * self.cell_size - 5,
                fill="green", outline="black", tags=("goal", f"{x},{y}")
            )
            self.canvas.create_text(
                x * self.cell_size + self.cell_size // 2,
                y * self.cell_size + self.cell_size // 2,
                text=str(i+1), fill="white", font=("Arial", 12, "bold")
            )

    def on_canvas_click(self, event):
        if not self.problem or len(self.problem.goal_states) < 2:
            return

        x = event.x // self.cell_size
        y = event.y // self.cell_size

        for goal in self.problem.goal_states:
            if goal == (x, y):
                self.selected_goal = goal
                self.highlight_selected_goal()
                return

    def highlight_selected_goal(self):
        self.canvas.delete("selected_goal")
        if self.selected_goal:
            x, y = self.selected_goal
            self.canvas.create_rectangle(
                x * self.cell_size + 2, y * self.cell_size + 2,
                (x + 1) * self.cell_size - 2, (y + 1) * self.cell_size - 2,
                outline="blue", width=3, tags="selected_goal"
            )

    def run_single_goal_algorithm(self):
        if not self.problem:
            messagebox.showerror("Error", "No problem loaded")
            return

        algorithm = self.algorithm.get()
        self.run_single_goal_search(algorithm)

    def run_multi_goal_algorithm(self):
        if not self.problem:
            messagebox.showerror("Error", "No problem loaded")
            return

        self.run_multi_goal_search()

    def run_single_goal_search(self, algorithm):
        if len(self.problem.goal_states) > 1 and not self.selected_goal:
            messagebox.showerror("Error", "Please select a goal before running the algorithm")
            return

        search_method = getattr(self.problem, algorithm.lower())
        
        self.draw_grid()  # Reset grid
        self.highlight_selected_goal()
        self.update()

        if self.selected_goal:
            original_goals = self.problem.goal_states
            self.problem.goal_states = [self.selected_goal]

        if algorithm == 'IDA_STAR':
            goal, nodes, path = self.problem.ida_star()
        else:
            goal, nodes, path = search_method()
        
        self.visualize_search(self.problem.get_visited_nodes(), path)

        if self.selected_goal:
            self.problem.goal_states = original_goals

        if goal:
            self.status_bar.configure(text=f"Goal found at {goal}. Nodes created: {nodes}")
            messagebox.showinfo("Result", f"Goal found at {goal}.\nNodes created: {nodes}\nPath: {' -> '.join(path)}")
        else:
            self.status_bar.configure(text=f"Unable to reach goal. Nodes created: {nodes}")
            messagebox.showinfo("Result", f"Unable to reach goal.\nNodes created: {nodes}")

    def run_multi_goal_search(self):
        self.draw_grid()  # Reset grid
        self.update()

        path, nodes_created, actions = self.problem.astar_all_goals()
        self.visualize_search(self.problem.get_visited_nodes(), actions)

        if path:
            self.status_bar.configure(text=f"All goals reached. Nodes created: {nodes_created}")
            messagebox.showinfo("Result", f"All goals reached.\nNodes created: {nodes_created}\nPath: {' -> '.join(actions)}")
        else:
            self.status_bar.configure(text=f"Unable to reach all goals. Nodes created: {nodes_created}")
            messagebox.showinfo("Result", f"Unable to reach all goals.\nNodes created: {nodes_created}")

    def visualize_search(self, visited_nodes: set[Tuple[int, int]], path: List[str]):
        for node in visited_nodes:
            x, y = node
            self.canvas.create_rectangle(
                x * self.cell_size + 2, y * self.cell_size + 2,
                (x + 1) * self.cell_size - 2, (y + 1) * self.cell_size - 2,
                fill="lightblue", outline=""
            )
            self.update()
            time.sleep(self.delay / (1000 * self.animation_speed.get()))  # Adjust delay based on animation speed

        # Animate robot moving to goal and change block color
        current = self.problem.initial_state
        for action in path:
            next_state = self.get_next_state(current, action)
            self.canvas.move("robot",
                (next_state[0] - current[0]) * self.cell_size,
                (next_state[1] - current[1]) * self.cell_size
            )
            self.canvas.create_rectangle(
                current[0] * self.cell_size + 2, current[1] * self.cell_size + 2,
                (current[0] + 1) * self.cell_size - 2, (current[1] + 1) * self.cell_size - 2,
                fill="yellow", outline=""
            )
            current = next_state
            self.update()
            time.sleep(self.delay / (1000 * self.animation_speed.get()))

        # Highlight the final goal state
        self.canvas.create_rectangle(
            current[0] * self.cell_size + 2, current[1] * self.cell_size + 2,
            (current[0] + 1) * self.cell_size - 2, (current[1] + 1) * self.cell_size - 2,
            fill="yellow", outline=""
        )

        # Display arrows after reaching the final goal
        current = self.problem.initial_state
        for action in path:
            next_state = self.get_next_state(current, action)
            self.canvas.create_line(
                current[0] * self.cell_size + self.cell_size // 2,
                current[1] * self.cell_size + self.cell_size // 2,
                next_state[0] * self.cell_size + self.cell_size // 2,
                next_state[1] * self.cell_size + self.cell_size // 2,
                fill="orange", width=3, arrow=tk.LAST
            )
            current = next_state
            self.update()
            time.sleep(self.delay / (1000 * self.animation_speed.get()))

    def get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        x, y = state
        if action == 'up':
            return (x, y - 1)
        elif action == 'down':
            return (x, y + 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        else:
            raise ValueError(f"Invalid action: {action}")

if __name__ == "__main__":
    app = RobotNavigationGUI()
    app.mainloop()