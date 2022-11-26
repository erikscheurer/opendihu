import sys, os
import timeit
import argparse
import importlib
import distutils.util

# parse rank arguments
rank_no = int(sys.argv[-2])
n_ranks = int(sys.argv[-1])

# add variables subfolder to python path where the variables script is located
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path)
sys.path.insert(0, os.path.join(script_path,'variables'))

import variables              # file variables.py, defined default values for all parameters, you can set the parameters there

# if first argument contains "*.py", it is a custom variable definition file, load these values
if ".py" in sys.argv[0]:
  variables_path_and_filename = sys.argv[0]
  variables_path,variables_filename = os.path.split(variables_path_and_filename)  # get path and filename
  sys.path.insert(0, os.path.join(script_path,variables_path))                    # add the directory of the variables file to python path
  variables_module,_ = os.path.splitext(variables_filename)                       # remove the ".py" extension to get the name of the module

  if rank_no == 0:
    print("Loading variables from \"{}\".".format(variables_path_and_filename))

  custom_variables = importlib.import_module(variables_module, package=variables_filename)    # import variables module
  variables.__dict__.update(custom_variables.__dict__)
  sys.argv = sys.argv[1:]     # remove first argument, which now has already been parsed

# define command line arguments
mbool = lambda x:bool(distutils.util.strtobool(x))   # function to parse bool arguments
parser = argparse.ArgumentParser(description='fibers_emg')
parser.add_argument('--scenario_name',                       help='The name to identify this run in the log.',            default=variables.scenario_name)
parser.add_argument('--n_subdomains', nargs=3,               help='Number of subdomains in x,y,z direction.',             type=int)
parser.add_argument('--n_subdomains_x', '-x',                help='Number of subdomains in x direction.',                 type=int, default=variables.n_subdomains_x)
parser.add_argument('--n_subdomains_y', '-y',                help='Number of subdomains in y direction.',                 type=int, default=variables.n_subdomains_y)
parser.add_argument('--n_subdomains_z', '-z',                help='Number of subdomains in z direction.',                 type=int, default=variables.n_subdomains_z)
parser.add_argument('--diffusion_solver_type',               help='The solver for the diffusion.',                        default=variables.diffusion_solver_type, choices=["gmres","cg","lu","gamg","richardson","chebyshev","cholesky","jacobi","sor","preonly"])
parser.add_argument('--diffusion_preconditioner_type',       help='The preconditioner for the diffusion.',                default=variables.diffusion_preconditioner_type, choices=["jacobi","sor","lu","ilu","gamg","none"])
parser.add_argument('--diffusion_solver_reltol',             help='Ralative tolerance for diffusion solver',              type=float, default=variables.diffusion_solver_reltol)
parser.add_argument('--diffusion_solver_maxit',              help='Maximum number of iterations for diffusion solver',    type=int, default=variables.diffusion_solver_maxit)
parser.add_argument('--potential_flow_solver_type',          help='The solver for the potential flow (non-spd matrix).',  default=variables.potential_flow_solver_type, choices=["gmres","cg","lu","gamg","richardson","chebyshev","cholesky","jacobi","sor","preonly"])
parser.add_argument('--potential_flow_preconditioner_type',  help='The preconditioner for the potential flow.',           default=variables.potential_flow_preconditioner_type, choices=["jacobi","sor","lu","ilu","gamg","none"])
parser.add_argument('--potential_flow_solver_maxit',         help='Maximum number of iterations for potential flow solver', type=int, default=variables.potential_flow_solver_maxit)
parser.add_argument('--potential_flow_solver_reltol',        help='Relative tolerance for potential flow solver',         type=float, default=variables.potential_flow_solver_reltol)
parser.add_argument('--emg_solver_type',                     help='The solver for the static bidomain.',                  default=variables.emg_solver_type)
#parser.add_argument('--emg_solver_type',                    help='The solver for the static bidomain.',                  default=variables.emg_solver_type, choices=["gmres","cg","lu","gamg","richardson","chebyshev","cholesky","jacobi","sor","preonly"])
parser.add_argument('--emg_preconditioner_type',             help='The preconditioner for the static bidomain.',          default=variables.emg_preconditioner_type, choices=["jacobi","sor","lu","ilu","gamg","none"])
parser.add_argument('--emg_solver_maxit',                    help='Maximum number of iterations for activation solver',   type=int, default=variables.emg_solver_maxit)
parser.add_argument('--emg_solver_reltol',                   help='Relative tolerance for activation solver',             type=float, default=variables.emg_solver_reltol)
parser.add_argument('--emg_solver_abstol',                   help='Absolute tolerance for activation solver',             type=float, default=variables.emg_solver_abstol)
parser.add_argument('--emg_initial_guess_nonzero',           help='If the initial guess for the emg linear system should be set to the previous solution.', default=variables.emg_initial_guess_nonzero, action='store_true')
parser.add_argument('--paraview_output',                     help='Enable the paraview output writer.',          default=variables.paraview_output, action='store_true')
parser.add_argument('--python_output',                       help='Enable the python output writer.',                     default=variables.python_output, action='store_true')
parser.add_argument('--adios_output',                        help='Enable the MegaMol/ADIOS output writer.',          default=variables.adios_output, action='store_true')
parser.add_argument('--cellml_file',                         help='The filename of the file that contains the cellml model.', default=variables.cellml_file)
parser.add_argument('--fiber_distribution_file',             help='The filename of the file that contains the MU firing times.', default=variables.fiber_distribution_file)
parser.add_argument('--firing_times_file',                   help='The filename of the file that contains the cellml model.', default=variables.firing_times_file)
parser.add_argument('--stimulation_frequency',               help='Stimulations per ms. Each stimulation corresponds to one line in the firing_times_file.', default=variables.stimulation_frequency)
parser.add_argument('--end_time', '--tend', '-t',            help='The end simulation time.',                             type=float, default=variables.end_time)
parser.add_argument('--output_timestep',                     help='The timestep for writing outputs.',                    type=float, default=variables.output_timestep)
parser.add_argument('--output_timestep_fibers',              help='The timestep for writing fiber outputs.',              type=float, default=variables.output_timestep_fibers)
parser.add_argument('--output_timestep_3D_emg',              help='The timestep for writing 3D emg outputs.',             type=float, default=variables.output_timestep_emg)
parser.add_argument('--output_timestep_surface',             help='The timestep for writing 2D emg surface outputs.',     type=float, default=variables.output_timestep_surface)
parser.add_argument('--dt_0D',                               help='The timestep for the 0D model.',                       type=float, default=variables.dt_0D)
parser.add_argument('--dt_1D',                               help='The timestep for the 1D model.',                       type=float, default=variables.dt_1D)
parser.add_argument('--dt_splitting',                        help='The timestep for the splitting.',                      type=float, default=variables.dt_splitting_0D1D)
parser.add_argument('--dt_bidomain',                         help='The timestep for the 3D bidomain model.', type=float, default=variables.dt_bidomain)
parser.add_argument('--dt_elasticity',                       help='The timestep for the elasticity model.', type=float, default=variables.dt_elasticity)
parser.add_argument('--optimization_type',                   help='The optimization_type in the cellml adapter.',         default=variables.optimization_type, choices=["vc", "simd", "openmp", "gpu"])
parser.add_argument('--approximate_exponential_function',    help='Approximate the exp function by a Taylor series',      type=mbool, default=variables.approximate_exponential_function)
parser.add_argument('--maximum_number_of_threads',           help='If optimization_type is "openmp", the max thread number, 0=all.', type=int, default=variables.maximum_number_of_threads)
parser.add_argument('--use_aovs_memory_layout',              help='If optimization_type is "vc", whether to use AoVS memory layout.', type=mbool, default=variables.use_aovs_memory_layout)
parser.add_argument('--disable_firing_output',               help='Disables the initial list of fiber firings.',          default=variables.disable_firing_output, action='store_true')
parser.add_argument('--enable_surface_emg',                  help='Enable the surface emg output writer.',                type=mbool, default=variables.enable_surface_emg)
parser.add_argument('--fast_monodomain_solver_optimizations',help='Enable the optimizations for fibers.',                 type=mbool, default=variables.fast_monodomain_solver_optimizations)
parser.add_argument('--enable_weak_scaling',                 help='Disable optimization for not stimulated fibers.',      default=False, action='store_true')
parser.add_argument('--v',                                   help='Enable full verbosity in c++ code')
parser.add_argument('-v',                                    help='Enable verbosity level in c++ code',                   action="store_true")
parser.add_argument('-vmodule',                              help='Enable verbosity level for given file in c++ code')
parser.add_argument('-pause',                                help='Stop at parallel debugging barrier',                   action="store_true")
# parameter for the 3D mesh generation
parser.add_argument('--mesh3D_sampling_stride', nargs=3,     help='Stride to select the mesh points in x, y and z direction.', type=int, default=None)
parser.add_argument('--mesh3D_sampling_stride_x',            help='Stride to select the mesh points in x direction.',     type=int, default=variables.sampling_stride_x)
parser.add_argument('--mesh3D_sampling_stride_y',            help='Stride to select the mesh points in y direction.',     type=int, default=variables.sampling_stride_y)
parser.add_argument('--mesh3D_sampling_stride_z',            help='Stride to select the mesh points in z direction.',     type=int, default=variables.sampling_stride_z)

# parse command line arguments and assign values to variables module
args, other_args = parser.parse_known_args(args=sys.argv[:-2], namespace=variables)
if len(other_args) != 0 and rank_no == 0:
    print("Warning: These arguments were not parsed by the settings python file\n  " + "\n  ".join(other_args), file=sys.stderr)



# initialize some dependend variables
if variables.n_subdomains is not None:
  variables.n_subdomains_x = variables.n_subdomains[0]
  variables.n_subdomains_y = variables.n_subdomains[1]
  variables.n_subdomains_z = variables.n_subdomains[2]

variables.n_subdomains = variables.n_subdomains_x*variables.n_subdomains_y*variables.n_subdomains_z
if variables.enable_weak_scaling:
  variables.fast_monodomain_solver_optimizations = False

# 3D mesh resolution
if variables.mesh3D_sampling_stride is not None:
    variables.mesh3D_sampling_stride_x = variables.mesh3D_sampling_stride[0]
    variables.mesh3D_sampling_stride_y = variables.mesh3D_sampling_stride[1]
    variables.mesh3D_sampling_stride_z = variables.mesh3D_sampling_stride[2]
variables.sampling_stride_x = variables.mesh3D_sampling_stride_x
variables.sampling_stride_y = variables.mesh3D_sampling_stride_y
variables.sampling_stride_z = variables.mesh3D_sampling_stride_z

# automatically initialize partitioning if it has not been set
if n_ranks != variables.n_subdomains:

  # create all possible partitionings to the given number of ranks
  optimal_value = n_ranks**(1/3)
  possible_partitionings = []
  for i in range(1,n_ranks+1):
    for j in range(1,n_ranks+1):
      if i*j <= n_ranks and n_ranks % (i*j) == 0:
        k = int(n_ranks / (i*j))
        performance = (k-optimal_value)**2 + (j-optimal_value)**2 + 1.1*(i-optimal_value)**2
        possible_partitionings.append([i,j,k,performance])

  # if no possible partitioning was found
  if len(possible_partitionings) == 0:
    if rank_no == 0:
      print("\n\n\033[0;31mError! Number of ranks {} does not match given partitioning {} x {} x {} = {} and no automatic partitioning could be done.\n\n\033[0m".format(n_ranks, variables.n_subdomains_x, variables.n_subdomains_y, variables.n_subdomains_z, variables.n_subdomains_x*variables.n_subdomains_y*variables.n_subdomains_z))
    quit()

  # select the partitioning with the lowest value of performance which is the best
  lowest_performance = possible_partitionings[0][3]+1
  for i in range(len(possible_partitionings)):
    if possible_partitionings[i][3] < lowest_performance:
      lowest_performance = possible_partitionings[i][3]
      variables.n_subdomains_x = possible_partitionings[i][0]
      variables.n_subdomains_y = possible_partitionings[i][1]
      variables.n_subdomains_z = possible_partitionings[i][2]

# output information of run
if rank_no == 0:
  print("scenario_name: {},  n_subdomains: {} {} {},  n_ranks: {},  end_time: {}".format(variables.scenario_name, variables.n_subdomains_x, variables.n_subdomains_y, variables.n_subdomains_z, n_ranks, variables.end_time))
  print("dt_0D:           {:0.1e}, diffusion_solver_type:      {}".format(variables.dt_0D, variables.diffusion_solver_type))
  print("dt_1D:           {:0.1e}, potential_flow_solver_type: {}, approx. exp.: {}".format(variables.dt_1D, variables.potential_flow_solver_type, variables.approximate_exponential_function))
  print("dt_splitting:    {:0.1e}, emg_solver_type:            {}, emg_initial_guess_nonzero: {}".format(variables.dt_splitting_0D1D, variables.emg_solver_type, variables.emg_initial_guess_nonzero))
  print("dt_3D:           {:0.1e}, paraview_output: {}, optimization_type: {}{}".format(variables.dt_bidomain, variables.paraview_output, variables.optimization_type, " ({} threads)".format(variables.maximum_number_of_threads) if variables.optimization_type=="openmp" else " (AoVS)" if variables.optimization_type=="vc" and variables.use_aovs_memory_layout else " (SoVA)" if variables.optimization_type=="vc" and not variables.use_aovs_memory_layout else ""))
  print("output_timestep: {:0.1e}, surface: {:0.1e}, stimulation_frequency: {} 1/ms = {} Hz".format(variables.output_timestep, variables.output_timestep_surface, variables.stimulation_frequency, variables.stimulation_frequency*1e3))
  print("                          fast_monodomain_solver_optimizations: {}".format(variables.fast_monodomain_solver_optimizations))
  print("cellml_file:             {}".format(variables.cellml_file))
  print("fiber_distribution_file: {}".format(variables.fiber_distribution_file))
  print("firing_times_file:       {}".format(variables.firing_times_file))
  print("********************************************************************************")

  print("prefactor: sigma_eff/(Am*Cm) = {} = {} / ({}*{})".format(variables.Conductivity/(variables.Am*variables.Cm), variables.Conductivity, variables.Am, variables.Cm))

  # start timer to measure duration of parsing of this script
  t_start_script = timeit.default_timer()

# initialize all helper variables
from helper import *

variables.n_subdomains_xy = variables.n_subdomains_x * variables.n_subdomains_y
variables.n_fibers_total = variables.n_fibers_x * variables.n_fibers_y



#### set Dirichlet BC and Neumann BC for the free side of the muscle

[nx, ny, nz] = [elem + 1 for elem in variables.n_elements_muscle1]
[mx, my, mz] = [elem // 2 for elem in variables.n_elements_muscle1] # quadratic elements consist of 2 linear elements along each axis

variables.elasticity_dirichlet_bc = {}

k = 0 #free side of the muscle

for j in range(ny):
    for i in range(nx):
      variables.elasticity_dirichlet_bc[k*nx*ny + j*nx + i] = [None, None, 0.0, None,None,None] # displacement ux uy uz, velocity vx vy vz

for k in range(nz):
    variables.elasticity_dirichlet_bc[k*nx*ny] = [0.0, 0.0, None, None,None,None] # displacement ux uy uz, velocity vx vy vz

variables.elasticity_dirichlet_bc[0] = [0.0, 0.0, 0.0, None,None,None] # displacement ux uy uz, velocity vx vy vz

# Neumann BC: increasing traction

# variables.force = 20.0
# k = mz-1
# variables.elasticity_neumann_bc = [{"element": k*mx*my + j*mx + i, "constantVector": [0,0,variables.force], "face": "2+"} for j in range(my) for i in range(mx)]

# def update_neumann_bc(t):

#   # set new Neumann boundary conditions
#   factor = min(1, t/100)   # for t ∈ [0,100] from 0 to 1
#   elasticity_neumann_bc = [{
# 		"element": k*mx*my + j*mx + i, 
# 		"constantVector": [0,0, variables.force*factor], 		# force pointing to bottom
# 		"face": "2+",
#     "isInReferenceConfiguration": True
#   } for j in range(my) for i in range(mx)]

#   config = {
#     "inputMeshIsGlobal": True,
#     "divideNeumannBoundaryConditionValuesByTotalArea": True,            # if the given Neumann boundary condition values under "neumannBoundaryConditions" are total forces instead of surface loads and therefore should be scaled by the surface area of all elements where Neumann BC are applied
#     "neumannBoundaryConditions": elasticity_neumann_bc,
#   }
#   print("prescribed pulling force to bottom: {}".format(variables.force*factor))
#   return config

# meshes

# add neuron meshes
neuron_meshes = {
  # "motoneuronMesh": {
  #   # each muscle has the same number of muscle spindels
  #   "nElements" :         (2*variables.n_motoneurons)-1 if n_ranks == 1 else (2*variables.n_motoneurons)*n_ranks,  # the last dof is empty in parallel
  #   "physicalExtent":     0, # has no special meaning. only to seperate data points in paraview
  #   "physicalOffset":     0,
  #   "logKey":             "motoneuron",
  #   "inputMeshIsGlobal":  True,
  #   "nRanks":             n_ranks
  # },

  "muscle1Mesh": {
    "nElements" :         variables.n_elements_muscle1,
    "physicalExtent":     variables.muscle1_extent,
    "physicalOffset":     [0,0,0],
    "logKey":             "muscle1",
    "inputMeshIsGlobal":  True,
    "nRanks":             n_ranks
  },

  # needed for mechanics solver
  "muscle1Mesh_quadratic": {
    "nElements" :         [elems // 2 for elems in variables.n_elements_muscle1],
    "physicalExtent":     variables.muscle1_extent,
    "physicalOffset":     [0,0,0],
    "logKey":             "muscle1_quadratic",
    "inputMeshIsGlobal":  True,
    "nRanks":             n_ranks,
  }
}
variables.meshes.update(neuron_meshes)
variables.meshes.update(fiber_meshes)


def dbg(x, name=None):
  if name:
    print(name, end=': ')
  print(x)
  return x

# define the config dict
config = {
  "scenarioName":                   variables.scenario_name,    # scenario name which will appear in the log file
  "logFormat":                      "csv",                      # "csv" or "json", format of the lines in the log file, csv gives smaller files
  "solverStructureDiagramFile":     "solver_structure.txt",     # output file of a diagram that shows data connection between solvers
  "mappingsBetweenMeshesLogFile":   "out/" + variables.scenario_name + "/mappings_between_meshes.txt",
  "meta": {                 # additional fields that will appear in the log
    "partitioning": [variables.n_subdomains_x, variables.n_subdomains_y, variables.n_subdomains_z]
  },
  "Meshes":                variables.meshes,
  # meshes know their coordinates and the mapping happens automatically. We could also define other parameters such as a mapping tolerance
  "MappingsBetweenMeshes": {"muscle1_fiber{}".format(f) : ["muscle1Mesh", "muscle1Mesh_quadratic"] for f in range(variables.n_fibers_total)},
  "Solvers": {
    "diffusionTermSolver": {# solver for the implicit timestepping scheme of the diffusion time step
      "relativeTolerance":  variables.diffusion_solver_reltol,
      "absoluteTolerance":  1e-10,         # 1e-10 absolute tolerance of the residual
      "maxIterations":      variables.diffusion_solver_maxit,
      "solverType":         variables.diffusion_solver_type,
      "preconditionerType": variables.diffusion_preconditioner_type,
      "dumpFilename":       "",   # "out/dump_"
      "dumpFormat":         "matlab",
    },
    # "potentialFlowSolver": {# solver for the initial potential flow, that is needed to estimate fiber directions for the bidomain equation
    #   "relativeTolerance":  variables.potential_flow_solver_reltol,
    #   "absoluteTolerance":  1e-10,         # 1e-10 absolute tolerance of the residual
    #   "maxIterations":      variables.potential_flow_solver_maxit,
    #   "solverType":         variables.potential_flow_solver_type,
    #   "preconditionerType": variables.potential_flow_preconditioner_type,
    #   "dumpFilename":       "",
    #   "dumpFormat":         "matlab",
    #   "cycleType":          "cycleV",     # if the preconditionerType is "gamg", which cycle to use "cycleV" or "cycleW"
    #   "gamgType":           "agg",        # if the preconditionerType is "gamg", the type of the amg solver
    #   "nLevels":            25,           # if the preconditionerType is "gamg", the maximum number of levels
    # },
    # "muscularEMGSolver": {   # solver for the static Bidomain equation and the EMG
    #   "relativeTolerance":  variables.emg_solver_reltol,
    #   "absoluteTolerance":  variables.emg_solver_abstol,
    #   "maxIterations":      variables.emg_solver_maxit,
    #   "solverType":         variables.emg_solver_type,
    #   "preconditionerType": variables.emg_preconditioner_type,
    #   "dumpFilename":       "",
    #   "dumpFormat":         "matlab",
    # },
    "mechanicsSolver": {   # solver for the dynamic mechanics problem
      "relativeTolerance":   variables.linear_relative_tolerance,           # 1e-10 relative tolerance of the linear solver
      "absoluteTolerance":   variables.linear_absolute_tolerance,           # 1e-10 absolute tolerance of the residual of the linear solver
      "solverType":          variables.elasticity_solver_type,            # type of the linear solver
      "preconditionerType":  variables.elasticity_preconditioner_type,    # type of the preconditioner
      "maxIterations":       1e4,                                         # maximum number of iterations in the linear solver
      "snesMaxFunctionEvaluations": 1e8,                                  # maximum number of function iterations
      "snesMaxIterations":   variables.snes_max_iterations,               # maximum number of iterations in the nonlinear solver
      "snesRelativeTolerance": variables.snes_relative_tolerance,         # relative tolerance of the nonlinear solver
      "snesAbsoluteTolerance": variables.snes_absolute_tolerance,         # absolute tolerance of the nonlinear solver
      "snesLineSearchType": "l2",                                         # type of linesearch, possible values: "bt" "nleqerr" "basic" "l2" "cp" "ncglinear"
      "snesRebuildJacobianFrequency": variables.snes_rebuild_jacobian_frequency,    # how often the jacobian should be recomputed, -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time the Jacobian is built etc. -2 means rebuild at next chance but then never again
      "hypreOptions":        "",                                          # additional options for the hypre solvers could be given here
      "dumpFilename":        "",                                          # dump system matrix and right hand side after every solve
      "dumpFormat":          "matlab",                                    # default, ascii, matlab
    }
  },


  # connections of the slots, identified by slot name
  "connectedSlots": [
    # global slots only support named slots (connectedSlotsTerm1To2 also allows indices)

    # use global slot, because automatic connection of "Razumova/activestress" does not work for some reason
    # "Razumova/activestress" from CellML to Muscle contaction solver
    ("m1gout", "m1g_in"),
    ("m2gout", "m2g_in"),

    # lambda and derived values (by MapDofs) -> input of muscle splindel simulation
    ("ms0",    "ms_in0"),
    ("ms1",    "ms_in1"),
    ("ms2",    "ms_in2"),
    ("ms3",    "ms_in3"),
    ("ms4",    "ms_in4"),

    ("in_g",   "in_in"),
  ],


  "PreciceAdapterVolumeCoupling": {
    "timeStepOutputInterval": 1,
    "timestepWidth":          variables.dt_elasticity,
    'description':            "everything",
    "couplingEnabled":          True,                       # if the precice coupling is enabled, if not, it simply calls the nested solver, for debugging
    "endTimeIfCouplingDisabled": variables.end_time,        # if "couplingEnabled" is set to False, use this end time for the simulation
    "preciceConfigFilename":    "../precice_config.xml",    # the name of the precice configuration file
    "preciceParticipantName":   "MuscleContraction",               # the name of the participant in the precice configuration file
    "scalingFactor":            1.0,                        # scaling factor for the coupling data
    "outputOnlyConvergedTimeSteps": True,                   # if only time steps are written to file, where the coupling converged
    
    "preciceData": [
      {
        "mode":                 "read",                     # mode is one of "read" or "write"
        "preciceDataName":      "Traction",                 # name of the vector or scalar to transfer, as given in the precice xml settings file
        "preciceMeshName":      "MuscleContractionMesh",    # name of the precice coupling mesh, as given in the precice xml settings file
        "opendihuMeshName":     None,                       # extra specification of the opendihu mesh that is used for the initialization of the precice mapping. If None or "", the mesh of the field variable is used.
        "slotName":             "m1T",                       # name of the existing slot of the opendihu data connector to which this variable is associated to (only relevant if not isGeometryField)
        "isGeometryField":      False,                       # if this is the geometry field of the mesh
      },
      {
        "mode":                 "write",                    # mode is one of "read" or "write"
        "preciceDataName":      "Gamma",                    # name of the vector or scalar to transfer, as given in the precice xml settings file
        "preciceMeshName":      "MuscleContractionMesh",    # name of the precice coupling mesh, as given in the precice xml settings file
        "opendihuMeshName":     None,                       # extra specification of the opendihu mesh that is used for the initialization of the precice mapping. If None or "", the mesh of the field variable is used.
        "slotName":             "m1g_in",                      # name of the existing slot of the opendihu data connector to which this variable is associated to (only relevant if not isGeometryField)
        "isGeometryField":      False,                      # if this is the geometry field of the mesh
      },
    ],

      "MuscleContractionSolver": {
            "numberTimeSteps":              1,                         # only use 1 timestep per interval
            "timeStepOutputInterval":       1,
            "Pmax":                         variables.Pmax,            # maximum PK2 active stress
            "enableForceLengthRelation":    True,                      # if the factor f_l(λ_f) modeling the force-length relation (as in Heidlauf2013) should be multiplied. Set to false if this relation is already considered in the CellML model.
            "lambdaDotScalingFactor":       1.0,                       # scaling factor for the output of the lambda dot slot, i.e. the contraction velocity. Use this to scale the unit-less quantity to, e.g., micrometers per millisecond for the subcellular model.
            "slotNames":                    ["m1lda", "m1ldot", "m1g_in", "m1T", "m1ux", "m1uy", "m1uz"],  # slot names of the data connector slots: lambda, lambdaDot, gamma, traction
            "OutputWriter" : [
              {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/" + variables.scenario_name + "/muscle1_contraction", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles": True, "fileNumbering": "incremental"},
            ],
            "mapGeometryToMeshes":          ["muscle1Mesh"] + [key for key in fiber_meshes.keys() if "muscle1_fiber" in key],    # the mesh names of the meshes that will get the geometry transferred
            "reverseMappingOrder":          True,                      # if the mapping target->own mesh should be used instead of own->target mesh. This gives better results in some cases.
            "dynamic":                      variables.dynamic,                      # if the dynamic solid mechanics solver should be used, else it computes the quasi-static problem

            # the actual solid mechanics solver, this is either "DynamicHyperelasticitySolver" or "HyperelasticitySolver", depending on the value of "dynamic"
            "DynamicHyperelasticitySolver": {
              "timeStepWidth":              variables.dt_elasticity,           # time step width
              "durationLogKey":             "muscle1_duration_mechanics",               # key to find duration of this solver in the log file
              "timeStepOutputInterval":     1,                         # how often the current time step should be printed to console

              "materialParameters":         variables.material_parameters,  # material parameters of the Mooney-Rivlin material
              "density":                    variables.rho,             # density of the material
              "dampingFactor":              variables.damping_factor,  # factor for velocity dependent damping
              "displacementsScalingFactor": 1.0,                       # scaling factor for displacements, only set to sth. other than 1 only to increase visual appearance for very small displacements
              "residualNormLogFilename":    "out/"+variables.scenario_name+"/muscle1_log_residual_norm.txt",   # log file where residual norm values of the nonlinear solver will be written
              "useAnalyticJacobian":        True,                      # whether to use the analytically computed jacobian matrix in the nonlinear solver (fast)
              "useNumericJacobian":         False,                     # whether to use the numerically computed jacobian matrix in the nonlinear solver (slow), only works with non-nested matrices, if both numeric and analytic are enable, it uses the analytic for the preconditioner and the numeric as normal jacobian

              "dumpDenseMatlabVariables":   False,                     # whether to have extra output of matlab vectors, x,r, jacobian matrix (very slow)
              # if useAnalyticJacobian,useNumericJacobian and dumpDenseMatlabVariables all all three true, the analytic and numeric jacobian matrices will get compared to see if there are programming errors for the analytic jacobian

              # mesh
              "inputMeshIsGlobal":          True,                     # boundary conditions and initial values are given as global numbers (every process has all information)
              "meshName":                   "muscle1Mesh_quadratic",       # name of the 3D mesh, it is defined under "Meshes" at the beginning of this config
              "fiberMeshNames":             [],                       # fiber meshes that will be used to determine the fiber direction
              "fiberDirection":             [0,0,1],                  # if fiberMeshNames is empty, directly set the constant fiber direction, in element coordinate system

              # solving
              "solverName":                 "mechanicsSolver",         # name of the nonlinear solver configuration, it is defined under "Solvers" at the beginning of this config
              #"loadFactors":                [0.5, 1.0],                # load factors for every timestep
              "loadFactors":                [],                        # no load factors, solve problem directly
              "loadFactorGiveUpThreshold":  0.25,                       # a threshold for the load factor, when to abort the solve of the current time step. The load factors are adjusted automatically if the nonlinear solver diverged. If the load factors get too small, it aborts the solve.
              "scaleInitialGuess":          False,                     # when load stepping is used, scale initial guess between load steps a and b by sqrt(a*b)/a. This potentially reduces the number of iterations per load step (but not always).
              "nNonlinearSolveCalls":       1,                         # how often the nonlinear solve should be repeated

              # boundary and initial conditions
              "dirichletBoundaryConditions": variables.elasticity_dirichlet_bc,   # the initial Dirichlet boundary conditions that define values for displacements u and velocity v
              "neumannBoundaryConditions":   variables.elasticity_neumann_bc,     # Neumann boundary conditions that define traction forces on surfaces of elements
              "divideNeumannBoundaryConditionValuesByTotalArea": True,            # if the given Neumann boundary condition values under "neumannBoundaryConditions" are total forces instead of surface loads and therefore should be scaled by the surface area of all elements where Neumann BC are applied
              "updateDirichletBoundaryConditionsFunction": None,                  # muscle1_update_dirichlet_boundary_conditions_helper, function that updates the dirichlet BCs while the simulation is running
              "updateDirichletBoundaryConditionsFunctionCallInterval": 1,         # every which step the update function should be called, 1 means every time step
              "updateNeumannBoundaryConditionsFunction":   None,                    # function that updates the Neumann BCs while the simulation is running
              "updateNeumannBoundaryConditionsFunctionCallInterval": 1,           # every which step the update function should be called, 1 means every time step


              "initialValuesDisplacements":  [[0.0,0.0,0.0] for _ in range(variables.n_points_global)],     # the initial values for the displacements, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
              "initialValuesVelocities":     [[0.0,0.0,0.0] for _ in range(variables.n_points_global)],     # the initial values for the velocities, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
              "extrapolateInitialGuess":     True,                                # if the initial values for the dynamic nonlinear problem should be computed by extrapolating the previous displacements and velocities
              "constantBodyForce":           [0, 0, 0],       # a constant force that acts on the whole body, e.g. for gravity

              "dirichletOutputFilename":     "out/"+variables.scenario_name+"/muscle1_dirichlet_boundary_conditions",     # output filename for the dirichlet boundary conditions, set to "" to have no output
              "totalForceLogFilename":       "out/"+variables.scenario_name+"/muscle1_tendon_force.csv",              # filename of a log file that will contain the total (bearing) forces and moments at the top and bottom of the volume
              "totalForceLogOutputInterval":       10,                                  # output interval when to write the totalForceLog file

              # define which file formats should be written
              # 1. main output writer that writes output files using the quadratic elements function space. Writes displacements, velocities and PK2 stresses.
              "OutputWriter" : [

                # Paraview files
                {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/"+variables.scenario_name+"/muscle1_displacements", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},

                # Python callback function "postprocess"
                # responsible to model the tendon
                {"format": "PythonCallback", "outputInterval": 1, "callback": variables.muscle1_postprocess, "onlyNodalValues":True, "filename": "", "fileNumbering":'incremental'},
              ],
              # 2. additional output writer that writes also the hydrostatic pressure
              "pressure": {   # output files for pressure function space (linear elements), contains pressure values, as well as displacements and velocities
                "OutputWriter" : [
                  {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/"+variables.scenario_name+"/muscle1_pressure", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ]
              },
              # 3. additional output writer that writes virtual work terms
              "dynamic": {    # output of the dynamic solver, has additional virtual work values
                "OutputWriter" : [   # output files for displacements function space (quadratic elements)
                  {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/muscle1_dynamic", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                  {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/"+variables.scenario_name+"/muscle1_virtual_work", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ],
              },
              # 4. output writer for debugging, outputs files after each load increment, the geometry is not changed but u and v are written
              "LoadIncrements": {
                "OutputWriter" : [
                  {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/"+variables.scenario_name+"/muscle1_load_increments", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ]
              }  
            }   
          # }
      }
    # }
  }
  
}

# stop   timer and calculate how long parsing lasted
if rank_no == 0:
  t_stop_script = timeit.default_timer()
  print("Python config parsed in {:.1f}s.".format(t_stop_script - t_start_script))
