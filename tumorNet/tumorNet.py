import numpy as np
import random, math, os
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import argparse
import configparser
import scienceplots

# Define the style for plots
plt.style.use(['science', 'notebook', 'no-latex']) 
"""
TumorSimulator module (importable).

Usage:
    from tumor_simulator import TumorSimulator
    sim = TumorSimulator(...)
    sim.initialize(...)
    sim.step()
    sim.get_frame()  # for notebook display
    df = sim.to_dataframe()
"""
from dataclasses import dataclass
import math, random, csv
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image

@dataclass
class Cell:
    """
    Cell class representing a single cell in the lattice.
        Attributes:
            is_stem (bool): whether the cell is a stem cell
            proliferation_capacity (int): remaining divisions for non-stem cells
            ps (float): probability of symmetric stem cell division
            alpha (float): probability of apoptosis upon division attempt
            tc (float): time to next cell division
            mean_cycle (float): mean cell cycle duration
            sd_cycle (float): standard deviation of cell cycle duration
    """
    is_stem: bool
    proliferation_capacity: int
    ps: float
    alpha: float
    tc: float
    mean_cycle: float = 24.0
    sd_cycle: float = 2.0

    def advance_time(self, dt: float):
        """Advance the cell's division timer by dt."""
        self.tc = max(0.0, self.tc - dt)

    def generate_tc(self):
        """Generate a new division timer based on the cell's cycle parameters."""
        t = random.gauss(self.mean_cycle, self.sd_cycle)
        return max(0.1, t)

    def attempt_divide(self):
        """Check if the cell is ready to divide."""
        return self.tc <= 1e-8

class Lattice:
    def __init__(self, nx:int, ny:int, neighborhood:str='moore', boundary:str='reflective'):
        assert neighborhood in ('moore','von_neumann')
        assert boundary in ('periodic','reflective','dirichlet')
        self.nx = nx
        self.ny = ny
        self.neighborhood = neighborhood
        self.boundary = boundary
        self.grid = np.empty((nx, ny), dtype=object)

    def in_bounds(self, x, y):
        """Check if coordinates are within lattice bounds."""
        return 0 <= x < self.nx and 0 <= y < self.ny

    def wrap_coords(self, x, y):
        """Wrap coordinates for periodic boundaries."""
        if self.boundary == 'periodic':
            return x % self.nx, y % self.ny
        return x, y

    def get_neighbors(self, x, y):
        """
        Get the coordinates of the neighboring cells.
            Inputs:
                x (int): x-coordinate of the cell
                y (int): y-coordinate of the cell
            Outputs:
                coords (list of tuples): list of (i,j) coordinates of neighboring cells
        """
        # Select neighbor deltas based on neighborhood type
        if self.neighborhood == 'von_neumann':
            deltas = [(-1,0),(1,0),(0,-1),(0,1)]
        else:
            deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        coords=[]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if self.boundary=='periodic':
                nx, ny = self.wrap_coords(nx, ny) 
                coords.append((nx, ny))
            else:
                if self.in_bounds(nx, ny): 
                    coords.append((nx, ny))
        return coords

    def vacant_neighbors(self,x,y):
        """
        Get a list of vacant neighboring coordinates.
            Inputs:
                x (int): x-coordinate of the cell
                y (int): y-coordinate of the cell
            Outputs:
                vacancies (list of tuples): list of (i,j) coordinates of vacant neighbors
        """
        return [(i,j) for (i,j) in self.get_neighbors(x,y) if self.grid[i,j] is None]

    def place_cell(self,x,y,cell:Cell): 
        """Place a cell at the specified coordinates."""
        self.grid[x,y]=cell

    def remove_cell(self,x,y): 
        """Remove a cell from the specified coordinates."""
        self.grid[x,y]=None

    def iterate_cells_random_order(self):
        """
        Iterate over all occupied grid cells in random order.
            Output:
                coords (list of tuples): list of (x,y) coordinates of occupied cells 
                                         in random order
        """
        # Check occupied cells
        occ = np.vectorize(lambda c: c is not None)(self.grid)
        # Get their coordinates
        coords = list(zip(*np.nonzero(occ)))
        # Convert indices to int and shuffle
        coords = [(int(a),int(b)) for a,b in coords]
        random.shuffle(coords)

        return coords
    
    def counts(self):
        """Count total, stem, and non-stem cells in the lattice."""
        tot = stem = ns = 0
        for i in range(self.nx):
            for j in range(self.ny):
                c = self.grid[i,j]
                if c is not None:
                    tot += 1
                    if c.is_stem: 
                        stem += 1
                    else: 
                        ns += 1
        return {'total': tot,'stem': stem,'non_stem': ns}

class NutrientField:
    def __init__(self, nx, ny, D=1.0, decay=0.0, boundary='reflective', initial_value=1.0):
        self.nx=nx; self.ny=ny
        self.D=float(D); self.decay=float(decay)
        self.boundary=boundary
        self.n_iv = float(initial_value)
        self.c = np.full((nx,ny), float(initial_value), dtype=float)

    def laplacian(self, arr):
        """
        Compute the Laplacian of a 2D array.
            Inputs:
                arr (array, float): 2D array to compute the Laplacian of
            Outputs:
                lap (array, float): 2D array of the Laplacian
        """
        nx,ny=self.nx,self.ny
        lap=np.zeros_like(arr)
        for i in range(nx):
            for j in range(ny):
                center=arr[i,j]; s=0.0; count=0
                for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni,nj=i+di,j+dj
                    if self.boundary=='periodic':
                        ni%=nx; nj%=ny; s+=arr[ni,nj]; count+=1
                    else:
                        if 0<=ni<nx and 0<=nj<ny:
                            s += arr[ni,nj]; count+=1
                        else:
                            if self.boundary=='reflective':
                                ri=min(max(ni,0),nx-1); rj=min(max(nj,0),ny-1)
                                s += arr[ri,rj]; count+=1
                            elif self.boundary=='dirichlet':
                                s += 0.0; count+=1
                lap[i,j] = s - count*center
        return lap

    def rhs(self, c, uptake_map):
        """
        Right-hand side of the nutrient PDE.
            Inputs:
                c (array, float): current nutrient concentration
                uptake_map (array, float): 2D array of uptake rates
            Outputs:
                arr (array, float): RHS of the PDE
        """
        return self.D * self.laplacian(c) - self.decay*c - uptake_map

    def rk4_step(self, dt, uptake_map):
        """
        Run a single RK4 step.
            Inputs:
                dt (float): time step
                uptake_map (array, float): 2D array of uptake rates
            Outputs:
                None (updates self.c in place)
        """
        c0=self.c
        k1 = self.rhs(c0, uptake_map)
        k2 = self.rhs(c0 + 0.5*dt*k1, uptake_map)
        k3 = self.rhs(c0 + 0.5*dt*k2, uptake_map)
        k4 = self.rhs(c0 + dt*k3, uptake_map)
        self.c = c0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        np.maximum(self.c, 0.0, out=self.c)

    def advance(self, total_dt, uptake_map, substeps=5):
        """
        Advance the nutrient field by a given time subesteps using RK4.
            Inputs:
                total_dt (float): total time to advance
                uptake_map (array, float): 2D array of uptake rates
                substeps (int): number of RK4 substeps to perform
            Outputs:
                None (updates self.c in place)
        """
        if substeps<=0: 
            substeps=1

        h = total_dt/float(substeps)

        for _ in range(substeps):
            # Use RK4 to update nutrient field
            self.rk4_step(h, uptake_map)

class TumorSimulator:
    def __init__(self,
                 nx = 80, ny = 80,
                 neighborhood = 'moore', boundary = 'reflective',
                 dt = 1.0, steps = 240,
                 ps = 0.5, alpha = 0.0, prolif_capacity = 5,
                 mean_cycle = 24.0, sd_cycle = 2.0,
                 diff_constant = 1.0, decay = 0.0, initial_nutrient = 1.0,
                 uptake_nonstem = 0.02, uptake_stem = 0.01,
                 diffusion_substeps = 5, chemotaxis_beta = 3.0,
                 therapy_conf: dict = None,
                 seed: Optional[int] = None):

        # Define the diffusive constant
        D = diff_constant

        if seed is not None:
            random.seed(int(seed)); np.random.seed(int(seed))
        
        # lattice and nutrient field
        self.lattice = Lattice(nx, ny, neighborhood, boundary)
        self.nutrient = NutrientField(nx, ny, D, decay, boundary, initial_nutrient)

        # sim params
        self.dt = float(dt); self.steps = int(steps)
        self.default_ps = float(ps); self.default_alpha = float(alpha); self.default_prolif_capacity = int(prolif_capacity)
        self.mean_cycle = float(mean_cycle); self.sd_cycle = float(sd_cycle)

        # nutrient coupling params
        self.uptake_nonstem = float(uptake_nonstem); self.uptake_stem = float(uptake_stem)
        self.diffusion_substeps = int(diffusion_substeps); self.chemotaxis_beta = float(chemotaxis_beta)

        # therapy
        therapy_conf = therapy_conf or {}
        self.therapy_on = therapy_conf.get('therapy_on', False)
        self.therapy_start = float(therapy_conf.get('therapy_start', 0.0))
        self.therapy_duration = float(therapy_conf.get('therapy_duration', 0.0))
        self.therapy_period = float(therapy_conf.get('therapy_period', 0.0))
        self.therapy_kill_prob_nonstem = float(therapy_conf.get('therapy_kill_prob_nonstem', 0.5))
        self.therapy_kill_prob_stem = float(therapy_conf.get('therapy_kill_prob_stem', 0.1))
        self.therapy_reduce_prolif = int(therapy_conf.get('therapy_reduce_prolif', 0))
        self.therapy_apply_each_step = bool(therapy_conf.get('therapy_apply_each_step', True))

        # storage/visualization
        self.frames = []  # list of numpy arrays (RGB) for gif
        self.time_series = []
        self.cmap = {'empty':"white",'stem': "seagreen",'non_stem':"coral"}

    def initialize(self, init_seed='cluster', seed_coords:Optional[List[Tuple[int,int]]]=None, seed_count:int=1):
        """
        Initialize the tumor simulator.
            Inputs:
                init_seed (str): Initialization strategy ('single', 'cluster', 'random')
                seed_coords (list of tuples): Specific coordinates for cell placement
                seed_count (int): Number of cells to initialize
            Outputs:
                None
        """
        nx,ny = self.lattice.nx, self.lattice.ny
        if seed_coords:
            coords = seed_coords
        else:
            if init_seed=='single': 
                coords=[(nx//2, ny//2)]

            elif init_seed=='cluster':
                cx,cy = nx//2, ny//2
                coords = [(cx+dx, cy+dy) for dx in (-1,0,1) for dy in (-1,0,1)]

            elif init_seed=='random':
                coords=[]

                while len(coords)<seed_count:
                    x=random.randrange(nx); y=random.randrange(ny)

                    if self.lattice.grid[x,y] is None: 
                        coords.append((x,y))

            else: 
                raise ValueError('Unknown init_seed')

        for (x,y) in coords:
            if 0<=x<nx and 0<=y<ny:
                c = Cell(is_stem=True, proliferation_capacity=self.default_prolif_capacity, ps=self.default_ps,
                         alpha=self.default_alpha, tc=random.gauss(self.mean_cycle, self.sd_cycle),
                         mean_cycle=self.mean_cycle, sd_cycle=self.sd_cycle)
                self.lattice.place_cell(x,y,c)
        self.init_seed = init_seed

    # Utility 
    def compute_uptake_map(self):
        """
        Nutrient consumption map based on current lattice state.
            Output:
                arr(array, float): 2D array of uptake rates
        """
        arr = np.zeros((self.lattice.nx, self.lattice.ny), dtype=float)
        for i in range(self.lattice.nx):
            for j in range(self.lattice.ny):
                c = self.lattice.grid[i,j]
                if c is not None:
                    arr[i,j] = self.uptake_stem if c.is_stem else self.uptake_nonstem
        return arr

    def therapy_active_at_time(self,t):
        """
        Check if therapy is active at time t.
            Inputs:
                t (float): current time
            Outputs:
                active (bool): whether therapy is active
        """
        if not self.therapy_on: 
            return False
        
        if self.therapy_period>0:
            if t < self.therapy_start: 
                return False
            dt = t - self.therapy_start
            in_period = (dt % self.therapy_period)
            return in_period < self.therapy_duration
        else:
            return (t >= self.therapy_start) and (t < self.therapy_start + self.therapy_duration)

    def choose_division_site_with_chemotaxis(self, x,y, vacancies):
        """
        Choose a division site for the cell using chemotaxis.
            Inputs:
                x (int): x-coordinate of the cell
                y (int): y-coordinate of the cell
                vacancies (list of tuples): list of (i,j) coordinates of vacant neighbors
            Outputs:
                dest (tuple): (i,j) coordinates of chosen division site
        """
        # If there are no vacancies, return None
        if len(vacancies)==0: 
            return None
        
        # If chemotaxis is disabled, choose randomly
        if self.chemotaxis_beta==0.0: 
            return random.choice(vacancies)
        
        # Nutrient concentration at current cell
        cur = self.nutrient.c[x,y]
        weights=[]

        # Compute weights based on nutrient differences
        for (i,j) in vacancies:
            delta = self.nutrient.c[i,j] - cur

            # Compute chemotaxis weights with clamping to avoid overflow
            w = math.exp(max(min(self.chemotaxis_beta * delta, 50.0), -50.0))
            weights.append(w)

        tot=sum(weights)

        # Select site based on weights
        if tot<=0: 
            return random.choice(vacancies)
        
        r = random.random()*tot 
        cum=0.0

        for k,(i,j) in enumerate(vacancies):
            cum += weights[k]
            if r <= cum: 
                return (i,j)
        return vacancies[-1]

    def step(self, step_idx:int):
        """
        Perform a single simulation step.
            Inputs:
                step_idx (int): current step index
            Outputs:
                None (updates lattice and nutrient field in place)
        """
        t = step_idx * self.dt
        uptake = self.compute_uptake_map()
        self.nutrient.advance(self.dt, uptake, substeps=self.diffusion_substeps)
        coords = self.lattice.iterate_cells_random_order()
        for (x,y) in coords:
            # Get cell object
            cell = self.lattice.grid[x,y]
            if cell is None: 
                continue

            # Check therapy if active in this step
            therapy_active = self.therapy_active_at_time(t)
            if therapy_active and self.therapy_apply_each_step:
                kill_prob = self.therapy_kill_prob_stem if cell.is_stem else self.therapy_kill_prob_nonstem
                if random.random() < kill_prob:
                    # Kill the cell
                    self.lattice.remove_cell(x,y)
                    continue
                # Reduce proliferation capacity for non-stem cells
                if self.therapy_reduce_prolif>0 and not cell.is_stem:
                    cell.proliferation_capacity = max(0, cell.proliferation_capacity - self.therapy_reduce_prolif)

            # Update cell division timer
            cell.advance_time(self.dt)

            # Check if the cell is ready to divide
            if cell.attempt_divide():
                vac = self.lattice.vacant_neighbors(x,y)

                # If no vacancies, attempt migration
                if len(vac)==0:
                    self._attempt_migration(x,y,cell)
                else:
                    # Consider apoptosis upon division
                    if random.random() < cell.alpha:
                        self.lattice.remove_cell(x,y)
                        continue
                    # If not apoptotic, perform division
                    dest = self.choose_division_site_with_chemotaxis(x,y,vac)
                    self._perform_division(x,y,dest,cell)
            else:
                # Attempt migration with 20% probability
                if random.random() < 0.2:
                    self._attempt_migration(x,y,cell)

        cnts = self.lattice.counts()
        cnts.update({
            'time': t,
            'avg_nutrient': float(np.mean(self.nutrient.c)),
            'min_nutrient': float(np.min(self.nutrient.c)),
            'max_nutrient': float(np.max(self.nutrient.c)),
            'therapy_active': int(self.therapy_active_at_time(t)),
            'cell_state': self.lattice.grid.copy()
        })
        self.time_series.append(cnts)

    def _attempt_migration(self, x, y, cell):
        """
        Attempt to migrate the cell to a vacant neighboring position.
            Inputs:
                x (int): x-coordinate of the cell
                y (int): y-coordinate of the cell
                cell (Cell): the cell object to migrate
            Outputs:
                migrated (bool): whether the cell migrated
        """
        # Get vacant neighboring coordinates
        vac = self.lattice.vacant_neighbors(x,y)

        # Check if there are vacancies
        if not vac: 
            return False
        
        # Choose destination with chemotaxis
        dest = self.choose_division_site_with_chemotaxis(x, y, vac)

        # Migration probability
        k = len(vac)
        stay_prob = 1.0 / (1.0 + k) 

        # If the cell decides to stay, do nothing
        if random.random() < stay_prob:
            return False
        # Move the cell to the chosen destination
        self.lattice.place_cell(dest[0], dest[1], cell)
        # Remove the cell from its original position
        self.lattice.remove_cell(x, y)

        return True

    def _perform_division(self, x,y,dest,mother:Cell):
        """
        Perform cell division.
            Inputs:
                x (int): x-coordinate of the mother cell
                y (int): y-coordinate of the mother cell
                dest (tuple): (i,j) coordinates of the daughter cell's position
                mother (Cell): the mother cell object
            Outputs:
                None (updates lattice in place)
        """
        # Base case: no destination
        if dest is None: 
            return
        
        # non-stem
        if not mother.is_stem:
            # Update the proliferation capacity
            mother.proliferation_capacity -= 1

            # If no remaining capacity, remove the mother cell
            if mother.proliferation_capacity <= 0:
                self.lattice.remove_cell(x,y)
                return

            # Create the daughter cell (inherit mother's properties)
            daughter = Cell(is_stem=False, proliferation_capacity=mother.proliferation_capacity,
                            ps=mother.ps, alpha=mother.alpha, tc=mother.generate_tc(),
                            mean_cycle=mother.mean_cycle, sd_cycle=mother.sd_cycle)
            
            # Regenerate division timer
            mother.tc = mother.generate_tc()

            # Place the daughter cell in the lattice
            self.lattice.place_cell(dest[0], dest[1], daughter)
            return
        
        # stem-cells

        # Symmetric division
        if random.random() < mother.ps:
            daughter = Cell(is_stem=True, proliferation_capacity=mother.proliferation_capacity,
                            ps=mother.ps, alpha=mother.alpha, tc=mother.generate_tc(),
                            mean_cycle=mother.mean_cycle, sd_cycle=mother.sd_cycle)
            mother.tc = mother.generate_tc()
            self.lattice.place_cell(dest[0], dest[1], daughter)
        # Asymmetric division    
        else:
            daughter = Cell(is_stem=False, proliferation_capacity=mother.proliferation_capacity,
                            ps=mother.ps, alpha=mother.alpha, tc=mother.generate_tc(),
                            mean_cycle=mother.mean_cycle, sd_cycle=mother.sd_cycle)
            
            # Regenerate division timer
            mother.tc = mother.generate_tc()

            # Place the daughter cell in the lattice
            self.lattice.place_cell(dest[0], dest[1], daughter)

    def run(self, start_step = 0, n_steps = None, capture_frames = False, capture_every = 1, output_dir:Optional[str]=None):
        """
        Run the simulation for a specified number of steps.
            Inputs:
                start_step (int): step index to start from
                n_steps (int): number of steps to run (default: until self.steps)
                capture_frames (bool): whether to capture frames for visualization
                capture_every (int): interval of steps to capture frames
            Outputs:
                None (updates lattice, nutrient field, time series, and frames in place)
        """
        self.create_output_dir(output_dir=output_dir)

        if n_steps is None: 
            n_steps = self.steps
        
        end = start_step + n_steps
        
        for s in range(start_step, end):
            self.step(s)
            if capture_frames and (s % capture_every == 0):
                self.frames.append(self.get_frame_array(s))
        return

    def create_output_dir(self, output_dir:Optional[str]=None):
        """
        Create output directory for saving results.
            Inputs:
                output_dir (str): path to output directory (default: temporary directory)
            Outputs:
                output_dir (str): path to created output directory
        """      
        if output_dir is None:
            output_dir = "output"

        # Check if the directory exists, if not create it
        if os.path.isdir(output_dir):
            print(f"Directory '{output_dir}' already exists.")
        else:
            print(f"Directory '{output_dir}' has been created.")
            os.mkdir(output_dir)

        self.output_dir = output_dir

    def get_frame_array(self, step_idx:int):
        """ Generate a visualization frame as a numpy array.
            Inputs:
                step_idx (int): current step index 
            Outputs:
                arr (numpy array): RGB numpy array
        """
        # Determine lattice size
        nx,ny = self.lattice.nx, self.lattice.ny

        # Define therapy information
        if self.therapy_on:
            t_start = f"{self.therapy_start:.1f}"
            t_end = f"{self.therapy_start + self.therapy_duration:.1f}"
            therapy_status = f"{t_start} - {t_end}h"
        else:
            therapy_status = "Off"

        fig = plt.figure(figsize=(5,5), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(self.nutrient.c.T, cmap='Greys', vmin=0.0, vmax=self.nutrient.n_iv)

        # Empty lists for cell coordinates
        xs_s, ys_s, xs_n, ys_n = [], [], [], []

        # Collect cell coordinates
        for i in range(nx):
            for j in range(ny):
                c = self.lattice.grid[i,j]
                if c is not None:
                    if c.is_stem: 
                        xs_s.append(i); ys_s.append(j)
                    else: 
                        xs_n.append(i); ys_n.append(j)
        # Scatter plot cells
        if len(xs_n): 
            ax.scatter(xs_n, ys_n, s=13, marker='o', facecolors=[self.cmap['non_stem']], 
                       edgecolors='none', label = "Non-Stem Cells", alpha=0.6)
        if len(xs_s): 
            ax.scatter(xs_s, ys_s, s=13, marker='o', facecolors=[self.cmap['stem']], 
                       edgecolors='none', label = "Stem Cells", alpha=0.6)
        # Add a title and colorbar
        ax.set_title(f'Nutrient Field and Cancer Cells\n'+
                     r'$\rho_{sym}$='+f'{self.default_ps:.2f}, '+
                     r'$\alpha$='+f'{self.default_alpha:.2f}, '+
                     r'$\beta$='+f'{self.chemotaxis_beta:.2f}\n'+
                     r'$D$='+f'{self.nutrient.D:.2f}, '+
                     f'Therapy= {therapy_status}, '+
                     f'dt= {self.dt}h, '+
                     'IC=' + f'{self.init_seed}'.capitalize(), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add time annotation
        ax.text(0.01, 0.99, f"Time: {step_idx*self.dt:.1f}h", transform=ax.transAxes, va='top', fontsize=8, 
                bbox=dict(facecolor='white',alpha=0.7,edgecolor='none'))
        # Add label legend
        if len(xs_s) + len(xs_n) > 0:
            ax.legend(frameon = True, fontsize = 8, framealpha=0.7)
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Nutrient', fontsize=8)

        fig.canvas.draw()

        # Convert canvas to numpy array
        w,h = fig.canvas.get_width_height()

        # memory buffer of the RGBA image (remove the alpha channel)
        arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(h, w, 4)[..., :3]
        plt.close(fig)

        return arr
    
    def save_time_series_png(self,output_dir = None, filename='time_series.png'):
        """ Save the time series data as a PNG plot."""

        # Define therapy information
        if self.therapy_on:
            t_start = f"{self.therapy_start:.1f}"
            t_end = f"{self.therapy_start + self.therapy_duration:.1f}"
            therapy_status = f"{t_start} - {t_end}h"
        else:
            therapy_status = "Off"

        if not self.time_series:
            return
        # Extract data for plotting
        times = [row['time'] for row in self.time_series]
        totals = [row['total'] for row in self.time_series]
        stems = [row['stem'] for row in self.time_series]
        non_stems = [row['non_stem'] for row in self.time_series]
        avg_nutrients = [row['avg_nutrient'] for row in self.time_series]

        fig, ax1 = plt.subplots(figsize=(9,5), dpi=100)

        ax1.plot(times, totals, label='Total Cells', color='blue')
        ax1.plot(times, stems, label='Stem Cells', color='green')
        ax1.plot(times, non_stems, label='Non-Stem Cells', color='orange')
        ax1.set_xlabel('Time (h)', fontsize=11)
        ax1.set_ylabel('Cell Counts', fontsize=11)
        ax1.legend(frameon = True, loc = 6, fontsize=8, framealpha=0.7)

        ax2 = ax1.twinx()
        ax2.plot(times, avg_nutrients, label='Avg Nutrient', color='red', linestyle='--')
        ax2.set_ylabel('Average Nutrient', fontsize=11)
        ax2.legend(frameon = True, loc = 'upper right', fontsize=8, framealpha=0.7)

        plt.title('Tumor Growth and Nutrient Dynamics Over Time\n'+
                  r'$\rho_{sym}$='+f'{self.default_ps:.2f}, '+
                  r'$\alpha$='+f'{self.default_alpha:.2f}, '+
                  r'$\beta$='+f'{self.chemotaxis_beta:.2f}\n'+
                  r'$D$='+f'{self.nutrient.D:.2f}, '+
                  f'Therapy= {therapy_status}, '+
                  f'dt= {self.dt}h, '+
                  'IC=' + f'{self.init_seed}'.capitalize(), fontsize=12)
        if filename:
            plt.savefig(self.output_dir + "/" + filename)
            plt.close(fig)
            print(f"Time series plot saved to '{self.output_dir}/{filename}'")
        else:
            plt.show()

    def get_frame_pil(self, step_idx:int):
        """Generate a visualization frame as a PIL Image."""
        arr = self.get_frame_array(step_idx)
        return Image.fromarray(arr)

    def save_gif(self, output_dir=None, filename='tumor.gif', fps=6):
        """ Save the captured frames as a GIF file."""

        if not self.frames:
            # lazily generate frames from time_series length
            for s in range(len(self.time_series)):
                self.frames.append(self.get_frame_array(s))
        imageio.mimsave(self.output_dir + "/" + filename, self.frames, fps=fps)
        print(f"GIF saved to '{self.output_dir}/{filename}'")

    def to_dataframe(self):
        """ Convert the time series data to a pandas DataFrame."""
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("pandas required for to_dataframe()") from e
        return pd.DataFrame(self.time_series)

    def save_time_series_csv(self, output_dir=None, filename='counts.csv'):
        """Save the time series data to a CSV file."""

        if not self.time_series:
            return
        keys = ["time","total","stem","non_stem","avg_nutrient","min_nutrient","max_nutrient","therapy_active"]
        with open(self.output_dir + "/" + filename,'w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.time_series:
                writer.writerow({k:row.get(k,'') for k in keys})
        print(f"Time series data saved to '{self.output_dir}/{filename}'")

    def reset_frames(self):
        self.frames = []

    def reset_time_series(self):
        self.time_series = []

# Helper function for parsing config values
def _parse_value(v):
    v = v.strip()
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except ValueError:
        return v

# Helper function to print header and package information
def print_header(version):
    print("""
▗▄▄▄▖█  ▐▌▄▄▄▄   ▄▄▄   ▄▄▄ ▗▖  ▗▖▗▞▀▚▖   ■  
  █  ▀▄▄▞▘█ █ █ █   █ █    ▐▛▚▖▐▌▐▛▀▀▘▗▄▟▙▄▖
  █       █   █ ▀▄▄▄▀ █    ▐▌ ▝▜▌▝▚▄▄▖  ▐▌  
  █                        ▐▌  ▐▌       ▐▌  
                                        ▐▌                                                                                                                                                             
          """)
    print("TumorNet: A Cellular Automaton Tumor Growth Simulator")
    print(f"Version: {version}")
    print("Authors: Alan I. Palma, Sofia Feijóo")
    print("Date: 23 October 2025")
    print("")

def print_simulation_parameters(params: dict):
    for k, v in params.items():
        if k == 'therapy_on' and v == False:
            print(f"  {k}: {v}")
            break
        print(f"  {k}: {v}")
    print("")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TumorNet',
        description='''Tumor Growth Simulator''',
        epilog='Authors: Alan I. Palma, Sofia Feijóo\nDate: 23 October 2025'
    )
    
    # Version 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('-v', type=str, default='1.0.0', help='Version number')

    # Option for a ini file
    parser.add_argument('-c', '--config', type=str, default=None, help='Path to configuration file')

    # Initial conditions (defaults)
    parser.add_argument('-nx', type=int, default=80, help='Lattice size in x direction')
    parser.add_argument('-ny', type=int, default=80, help='Lattice size in y direction')
    parser.add_argument('--neighborhood', type=str, default='moore', help='Neighborhood type')
    parser.add_argument('--boundary', type=str, default='reflective', help='Boundary condition')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step')
    parser.add_argument('--steps', type=int, default=200, help='Number of simulation steps')
    parser.add_argument('--ps', type=float, default=0.5, help='Probability of survival')
    parser.add_argument('--alpha', type=float, default=0.0, help='Probability of apoptosis upon division')
    parser.add_argument('--prolif_capacity', type=int, default=5, help='Proliferation capacity')
    parser.add_argument('--mean_cycle', type=float, default=24.0, help='Mean cell cycle duration')
    parser.add_argument('--sd_cycle', type=float, default=2.0, help='Standard deviation of cell cycle duration')
    parser.add_argument('-D', '--diff_constant', type=float, default=1.0, help='Nutrient diffusion coefficient')
    parser.add_argument('--decay', type=float, default=0.0, help='Decay rate')
    parser.add_argument('--initial_nutrient', type=float, default=1.0, help='Initial nutrient concentration')
    parser.add_argument('--uptake_nonstem', type=float, default=0.02, help='Nutrient uptake rate for non-stem cells')
    parser.add_argument('--uptake_stem', type=float, default=0.01, help='Nutrient uptake rate for stem cells')
    parser.add_argument('--diffusion_substeps', type=int, default=5, help='Number of substeps for diffusion simulation')
    parser.add_argument('--chemotaxis_beta', type=float, default=3.0, help='Chemotaxis sensitivity')

    # Therapy configuration
    parser.add_argument('--therapy_on', type=bool, default=False, help='Enable therapy')
    parser.add_argument('--therapy_start', type=float, default=0.0, help='Start time of therapy')
    parser.add_argument('--therapy_duration', type=float, default=0.0, help='Duration of therapy')
    parser.add_argument('--therapy_period', type=float, default=0.0, help='Period of therapy')
    parser.add_argument('--therapy_kill_prob_nonstem', type=float, default=0.5, help='Kill probability for non-stem cells')
    parser.add_argument('--therapy_kill_prob_stem', type=float, default=0.1, help='Kill probability for stem cells')
    parser.add_argument('--therapy_reduce_prolif', type=int, default=0, help='Reduction in proliferation')
    parser.add_argument('--therapy_apply_each_step', type=bool, default=True, help='Apply therapy each step')

    # Initial seeding
    parser.add_argument('--init_seed', type=str, default='cluster', help='Initial seeding strategy')
    parser.add_argument('--seed_count', type=int, default=None, help='Number of seeds to place')
    parser.add_argument('--seed_coords', type=str, default=None, help='Coordinates for initial seeds')

    # Output options
    parser.add_argument('--save_gif', type=bool, default=True, help='Save GIF output')
    parser.add_argument('--save_time_series_png', type=bool, default=True, help='Save time series as PNG')
    parser.add_argument('--save_time_series_csv', type=bool, default=True, help='Save time series as CSV')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--output_gif', type=str, default='tumor_dynamics.gif', help='Output GIF file')
    parser.add_argument('--output_time_series_png', type=str, default='time_series.png', help='Output time series PNG file')
    parser.add_argument('--output_time_series_csv', type=str, default='counts.csv', help='Output time series CSV file')
    parser.add_argument('--frames_capture_every', type=int, default=1, help='Frames capture every N steps')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for output video')

    args = parser.parse_args()

    # Configure file parsing
    sim_params = {}
    therapy_conf = {}
    init_seed = {}
    output_params = {}
    

    if args.config and os.path.exists(args.config):
        mess = f"Loading configuration from '{args.config}'"
        config = configparser.ConfigParser()
        config.read(args.config)

        # Simulation parameters
        if 'SIMULATION' in config:
            for k, v in config['SIMULATION'].items():
                sim_params[k] = _parse_value(v)

        # Therapy configuration
        if 'THERAPY' in config:
            for k, v in config['THERAPY'].items():
                therapy_conf[k] = _parse_value(v)

        # Initial seeding
        if 'INIT_SEED' in config:
            for k, v in config['INIT_SEED'].items():
                init_seed[k] = _parse_value(v)

        # Output options
        if 'OUTPUT' in config:
            for k, v in config['OUTPUT'].items():
                output_params[k] = _parse_value(v)

    # Fallback defaults if no config provided
    sim_params.setdefault('nx', args.nx)
    sim_params.setdefault('ny', args.ny)
    sim_params.setdefault('neighborhood', args.neighborhood)
    sim_params.setdefault('boundary', args.boundary)
    sim_params.setdefault('dt', args.dt)
    sim_params.setdefault('steps', args.steps)
    sim_params.setdefault('ps', args.ps)
    sim_params.setdefault('alpha', args.alpha)
    sim_params.setdefault('prolif_capacity', args.prolif_capacity)
    sim_params.setdefault('mean_cycle', args.mean_cycle)
    sim_params.setdefault('sd_cycle', args.sd_cycle)
    sim_params.setdefault('diff_constant', args.diff_constant)
    sim_params.setdefault('decay', args.decay)
    sim_params.setdefault('initial_nutrient', args.initial_nutrient)
    sim_params.setdefault('uptake_nonstem', args.uptake_nonstem)
    sim_params.setdefault('uptake_stem', args.uptake_stem)
    sim_params.setdefault('diffusion_substeps', args.diffusion_substeps)
    sim_params.setdefault('chemotaxis_beta', args.chemotaxis_beta)

    # Also ensure therapy_conf exists (even if empty)
    therapy_conf.setdefault('therapy_on', args.therapy_on)
    therapy_conf.setdefault('therapy_start', args.therapy_start)
    therapy_conf.setdefault('therapy_duration', args.therapy_duration)
    therapy_conf.setdefault('therapy_period', args.therapy_period)
    therapy_conf.setdefault('therapy_kill_prob_nonstem', args.therapy_kill_prob_nonstem)
    therapy_conf.setdefault('therapy_kill_prob_stem', args.therapy_kill_prob_stem)
    therapy_conf.setdefault('therapy_reduce_prolif', args.therapy_reduce_prolif)
    therapy_conf.setdefault('therapy_apply_each_step', args.therapy_apply_each_step)

    # Initial seeding
    init_seed.setdefault('init_seed', args.init_seed)
    init_seed.setdefault('seed_count', args.seed_count)

    if args.seed_coords:
        coords = args.seed_coords.split(';')
        seed_coords = []
        for coord in coords:
            part = coord.strip().strip('()')
            x_str, y_str = part.split(',')
            seed_coords.append((int(x_str), int(y_str)))
        init_seed.setdefault('seed_coords', seed_coords)

    # Output defaults
    output_params.setdefault('save_gif', args.save_gif)
    output_params.setdefault('save_time_series_png', args.save_time_series_png)
    output_params.setdefault('save_time_series_csv', args.save_time_series_csv)
    output_params.setdefault('output_gif', args.output_gif)
    output_params.setdefault('output_time_series_png', args.output_time_series_png)
    output_params.setdefault('output_time_series_csv', args.output_time_series_csv)
    output_params.setdefault('frames_capture_every', args.frames_capture_every)
    output_params.setdefault('fps', args.fps)
    output_params.setdefault('output_dir', args.output_dir)

    # Print Header and Parameters
    print_header(version=args.v)

    if mess:
        print(mess)
        print("")
    
    else:
        print("Using configuration from command line arguments.")
        print("")

    print('-'*11, "Simulation Parameters", '-'*11)
    print_simulation_parameters(sim_params)

    print('-'*11, "Therapy Configuration", '-'*11)
    print_simulation_parameters(therapy_conf)

    print('-'*11, "Output Parameters", '-'*11)
    print_simulation_parameters(output_params)

    sim = TumorSimulator(**sim_params, therapy_conf=therapy_conf)
    
    sim.initialize(init_seed=init_seed["init_seed"],
                   seed_coords=init_seed.get("seed_coords", None),
                   seed_count=init_seed.get("seed_count", None))

    capture_frames = output_params['save_gif']
    capture_every = output_params['frames_capture_every']

    print("Running simulation...")
    sim.run(capture_frames=capture_frames, capture_every=capture_every, output_dir=output_params['output_dir'])
    

    if capture_frames:
        print("Saving GIF...")
        sim.save_gif(filename=output_params['output_gif'],
                     fps=output_params['fps'])
    
    if output_params['save_time_series_png']:
        print("Saving time series PNG...")
        sim.save_time_series_png(filename=output_params['output_time_series_png'])

    if output_params['save_time_series_csv']:
        print("Saving time series CSV...")
        sim.save_time_series_csv(filename=output_params['output_time_series_csv'])

    print("Simulation completed.")