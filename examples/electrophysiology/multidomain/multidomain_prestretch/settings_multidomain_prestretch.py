# biceps in multidomain, with previous contraction to obtain a smaller geometry and a stretching operation on this new geometry

import numpy as np
import scipy.stats
import pickle
import sys,os 
import timeit
import argparse
import importlib
import struct
sys.path.insert(0, "..")

# own MPI rank no and number of MPI ranks
rank_no = (int)(sys.argv[-2])
n_ranks = (int)(sys.argv[-1])

# add variables subfolder to python path where the variables script is located
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path)
sys.path.insert(0, os.path.join(script_path,'variables'))

import variables              # file variables.py, defined default values for all parameters, you can set the parameters there  
from create_partitioned_meshes_for_settings import *   # file create_partitioned_meshes_for_settings with helper functions about own subdomain

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
else:
  if rank_no == 0:
    print("Warning: There is no variables file, e.g:\n ./multidomain_prestretch ../settings_prestretch.py shortened.py\n")
  exit(0)
  
# -------------------------------------------------------- begin parameters ---------------------------------------------------------
# -------------------------------------------------------- end parameters ------------------------------------------------------------
parser = argparse.ArgumentParser(description='multidomain_contraction')
parser.add_argument('--scenario_name',                       help='The name to identify this run in the log.',   default=variables.scenario_name)
parser.add_argument('--n_subdomains', nargs=3,               help='Number of subdomains in x,y,z direction.',    type=int)
parser.add_argument('--n_subdomains_x', '-x',                help='Number of subdomains in x direction.',        type=int, default=variables.n_subdomains_x)
parser.add_argument('--n_subdomains_y', '-y',                help='Number of subdomains in y direction.',        type=int, default=variables.n_subdomains_y)
parser.add_argument('--n_subdomains_z', '-z',                help='Number of subdomains in z direction.',        type=int, default=variables.n_subdomains_z)
parser.add_argument('--potential_flow_solver_type',          help='The solver for the potential flow (non-spd matrix).', default=variables.potential_flow_solver_type, choices=["gmres","cg","lu","gamg","richardson","chebyshev","cholesky","jacobi","sor","preonly"])
parser.add_argument('--potential_flow_preconditioner_type',  help='The preconditioner for the potential flow.',  default=variables.potential_flow_preconditioner_type, choices=["jacobi","sor","lu","ilu","gamg","none"])
parser.add_argument('--paraview_output',                     help='Enable the paraview output writer.',          default=variables.paraview_output, action='store_true')
parser.add_argument('--adios_output',                        help='Enable the MegaMol/ADIOS output writer.',          default=variables.adios_output, action='store_true')
parser.add_argument('--fiber_file',                          help='The filename of the file that contains the fiber data.', default=variables.fiber_file)
parser.add_argument('--firing_times_file',                   help='The filename of the file that contains the cellml model.', default=variables.firing_times_file)
parser.add_argument('--end_time', '--tend', '-t',            help='The end simulation time.',                    type=float, default=variables.end_time)
parser.add_argument('--output_timestep_multidomain',         help='The timestep for writing outputs.',           type=float, default=variables.output_timestep_multidomain)
parser.add_argument('--multidomain_solver_type',             help='Solver for multidomain system.',              default=variables.multidomain_solver_type)
parser.add_argument('--multidomain_preconditioner_type',     help='Preconditioner for multidomain system.',      default=variables.multidomain_preconditioner_type)
parser.add_argument('--multidomain_relative_tolerance',      help='relative residual tol. for multidomain solver.',           type=float, default=variables.multidomain_relative_tolerance)
parser.add_argument('--multidomain_absolute_tolerance',      help='absolute residual tol. for multidomain solver.',           type=float, default=variables.multidomain_absolute_tolerance)
parser.add_argument('--theta',                               help='parameter of Crank-Nicholson in multidomain system.',      type=float, default=variables.theta)
parser.add_argument('--use_lumped_mass_matrix',              help='If the formulation with lumbed mass matrix should be used.',           default=variables.use_lumped_mass_matrix, action='store_true')
parser.add_argument('--use_symmetric_preconditioner_matrix', help='If the preconditioner matrix should be symmetric (only diagnoal part of system matrix).',  default=variables.use_symmetric_preconditioner_matrix, action='store_true')
parser.add_argument('--initial_guess_nonzero',               help='If the initial guess to the linear solver should be the last solution.',  default=variables.initial_guess_nonzero, action='store_true')

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

# automatically initialize partitioning if it has not been set
if n_ranks != variables.n_subdomains:
  
  # create all possible partitionings to the given number of ranks
  optimal_value = n_ranks**(1/3)
  possible_partitionings = []
  for i in range(1,n_ranks+1):
    for j in [1]:
      if i*j <= n_ranks and n_ranks % (i*j) == 0:
        k = (int)(n_ranks / (i*j))
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
  print("dt_0D:           {:0.0e}    multidomain solver:         {}, lumped mass matrix: {}".format(variables.dt_0D, variables.multidomain_solver_type, variables.use_lumped_mass_matrix))
  print("dt_multidomain:  {:0.0e}    multidomain preconditioner: {}, symmetric precond.: {}".format(variables.dt_multidomain, variables.multidomain_preconditioner_type, variables.use_symmetric_preconditioner_matrix))
  print("dt_splitting:    {:0.0e}    theta: {}, solver tolerances, abs: {}, rel: {}".format(variables.dt_splitting, variables.theta, variables.multidomain_absolute_tolerance, variables.multidomain_relative_tolerance))
  print("dt_elasticity:   {:0.0e}    elasticity solver: {}, preconditioner: {}".format(variables.dt_elasticity, variables.elasticity_solver_type, variables.elasticity_preconditioner_type))
  print("fiber_file:              {}".format(variables.fiber_file))
  print("fat_mesh_file:           {}".format(variables.fat_mesh_file))
  print("cellml_file:             {}".format(variables.cellml_file))
  print("firing_times_file:       {}".format(variables.firing_times_file))
  print("********************************************************************************")
  
  # start timer to measure duration of parsing of this script  
  t_start_script = timeit.default_timer()
    
# initialize all helper variables
from helper import *
  
# settings for the multidomain solver
multidomain_solver = {
  "timeStepWidth":                    variables.dt_multidomain,             # time step width of the subcellular problem
  "endTime":                          variables.end_time,                   # end time, this is not relevant because it will be overridden by the splitting scheme
  "timeStepOutputInterval":           1,                                  # how often the output timestep should be printed
  "durationLogKey":                   "duration_multidomain",               # key for duration in log.csv file
  "slotNames":                        ["vm_old", "vm_new", "g_mu", "g_tot"],  # names of the data connector slots, maximum length per name is 10 characters. g_mu is gamma (active stress) of the compartment, g_tot is the total gamma
  
  # material parameters for the compartments
  "nCompartments":                    variables.n_compartments,             # number of compartments
  "compartmentRelativeFactors":       variables.relative_factors.tolist(),  # list of lists of (the factors for all dofs), because "inputIsGlobal": True, this contains the global dofs
  "inputIsGlobal":                    True,                                 # if values and dofs correspond to the global numbering
  "am":                               [variables.get_am(mu_no) for mu_no in range(variables.n_compartments)],   # Am parameter for every motor unit (ration of surface to volume of fibers)
  "cm":                               [variables.get_cm(mu_no) for mu_no in range(variables.n_compartments)],   # Cm parameter for every motor unit (capacitance of the cellular membrane)
  
  # solver options
  "solverName":                       "multidomainLinearSolver",            # reference to the solver used for the global linear system of the multidomain eq.
  "alternativeSolverName":            "multidomainAlternativeLinearSolver", # reference to the alternative solver, which is used when the normal solver diverges
  "subSolverType":                    "gamg",                               # sub solver when block jacobi preconditioner is used
  "subPreconditionerType":            "none",                               # sub preconditioner when block jacobi preconditioner is used
  "hypreOptions":                     "-pc_hypre_boomeramg_strong_threshold 0.7",       # additional options if a hypre preconditioner is selected
  
  "theta":                            variables.theta,                      # weighting factor of implicit term in Crank-Nicolson scheme, 0.5 gives the classic, 2nd-order Crank-Nicolson scheme, 1.0 gives implicit euler
  "useLumpedMassMatrix":              variables.use_lumped_mass_matrix,     # which formulation to use, the formulation with lumped mass matrix (True) is more stable but approximative, the other formulation (False) is exact but needs more iterations
  "useSymmetricPreconditionerMatrix": variables.use_symmetric_preconditioner_matrix,    # if the diagonal blocks of the system matrix should be used as preconditioner matrix
  "initialGuessNonzero":              variables.initial_guess_nonzero,      # if the initial guess for the 3D system should be set as the solution of the previous timestep, this only makes sense for iterative solvers
  "enableFatComputation":             True,                                 # disabling the computation of the fat layer is only for debugging and speeds up computation. If set to False, the respective matrix is set to the identity
  "showLinearSolverOutput":           variables.show_linear_solver_output,  # if convergence information of the linear solver in every timestep should be printed, this is a lot of output for fast computations
  "updateSystemMatrixEveryTimestep":  False,                                # if this multidomain solver will update the system matrix in every first timestep, us this only if the geometry changed, e.g. by contraction
  "recreateLinearSolverInterval":     0,                                    # how often the Petsc KSP object (linear solver) should be deleted and recreated. This is to remedy memory leaks in Petsc's implementation of some solvers. 0 means disabled.
  "rescaleRelativeFactors":           True,                                 # if all relative factors should be rescaled such that max Σf_r = 1  
  "setDirichletBoundaryConditionPhiE":False,                                # (set to False) if the last dof of the extracellular space (variable phi_e) should have a 0 Dirichlet boundary co
  "setDirichletBoundaryConditionPhiB":False,                                # (set to False) if the last dof of the fat layer (variable phi_b) should have a 0 Dirichlet boundary condition. H
  "resetToAverageZeroPhiE":           True,                                 # if a constant should be added to the phi_e part of the solution vector after every solve, such that the average 
  "resetToAverageZeroPhiB":           True,                                 # if a constant should be added to the phi_b part of the solution vector after every solve, such that the average 
 

  "PotentialFlow": {
    "FiniteElementMethod" : {  
      "meshName":                     "3Dmesh",
      "solverName":                   "potentialFlowSolver",
      "prefactor":                    1.0,
      "dirichletBoundaryConditions":  variables.potential_flow_dirichlet_bc,
      "dirichletOutputFilename":      "out/"+variables.scenario_name+"/3_potential_flow_dirichlet_boundary_conditions",               # output filename for the dirichlet boundary conditions, set to "" to have no output
      "neumannBoundaryConditions":    [],
      "inputMeshIsGlobal":            True,
      "slotName":                     "",
    },
  },
  "Activation": {
    "FiniteElementMethod" : {  
      "meshName":                     "3Dmesh",
      "solverName":                   "multidomainLinearSolver",
      "prefactor":                    1.0,
      "inputMeshIsGlobal":            True,
      "dirichletBoundaryConditions":  {},
      "dirichletOutputFilename":      None,               # output filename for the dirichlet boundary conditions, set to "" to have no output
      "neumannBoundaryConditions":    [],
      "slotName":                     "",
      "diffusionTensor": [[      # sigma_i           # fiber direction is (1,0,0)
        8.93, 0, 0,
        0, 0.0, 0,
        0, 0, 0.0
      ]], 
      "extracellularDiffusionTensor": [[      # sigma_e
        6.7, 0, 0,
        0, 6.7, 0,
        0, 0, 6.7,
      ]],
    },
  },
  "Fat": {
    "FiniteElementMethod" : {  
      "meshName":                     "3DFatMesh",
      "solverName":                   "multidomainLinearSolver",
      "prefactor":                    0.4,
      "inputMeshIsGlobal":            True,
      "dirichletBoundaryConditions":  {},
      "dirichletOutputFilename":      None,               # output filename for the dirichlet boundary conditions, set to "" to have no output
      "neumannBoundaryConditions":    [],
      "slotName":                     "",
    },
  },
  
  "OutputWriter" : [
    {"format": "Paraview", "outputInterval": (int)(1./variables.dt_multidomain*variables.output_timestep_multidomain), "filename": "out/"+variables.scenario_name+"/3_multidomain", "binary": True, "fixedFormat": False, "combineFiles": True, "fileNumbering": "incremental"},
    #{"format": "ExFile", "filename": "out/fiber_"+str(i), "outputInterval": 1./dt_1D*output_timestep, "sphereSize": "0.02*0.02*0.02", "fileNumbering": "incremental"},
    #{"format": "PythonFile", "filename": "out/fiber_"+str(i), "outputInterval": int(1./dt_1D*output_timestep), "binary":True, "onlyNodalValues":True, "fileNumbering": "incremental"},
  ]
}
  
config = {
  "scenarioName":          variables.scenario_name,
  "logFormat":                      "csv",                                # "csv" or "json", format of the lines in the log file, csv gives smaller files
  "solverStructureDiagramFile":     "out/"+variables.scenario_name+"/solver_structure.txt",               # output file of a diagram that shows data connection between solvers
  "mappingsBetweenMeshesLogFile":   "out/"+variables.scenario_name+"/mappings_between_meshes_log.txt",    # log file for mappings 
  "meta": {                                                               # additional fields that will appear in the log
    "partitioning":         [variables.n_subdomains_x, variables.n_subdomains_y, variables.n_subdomains_z]
  },
  "connectedSlots": [
    ("stress", "g_mu"),     # stress output of CellML models -> gamma of the MU (input of MultidomainSolver)
    ("g_tot", "m_g_in")     # gamma total (gamma value of all compartments as computed by MultidomainSolver) ->  gamma_in of MuscleContractionSolver
  ],
  "Meshes":                variables.meshes,
  "MappingsBetweenMeshes": {
    "3Dmesh": [
       {"name": "3Dmesh_elasticity_quadratic+3DFatMesh_elasticity_quadratic", "xiTolerance": 0.5, "enableWarnings": True, "compositeUseOnlyInitializedMappings": True, "fixUnmappedDofs": True, "defaultValue": 0},
    ],
    "3DFatMesh":  [
       {"name": "3Dmesh_elasticity_quadratic+3DFatMesh_elasticity_quadratic", "xiTolerance": 0.2, "enableWarnings": True, "compositeUseOnlyInitializedMappings": True, "fixUnmappedDofs": True, "defaultValue": 0},
    ],
  },
  "Solvers": {
    "precontractionMechanicsSolver": {   # solver for the preprocessing contraction simulation (static mechanics problem)
      "relativeTolerance":   1e-5,           # 1e-10 relative tolerance of the linear solver
      "absoluteTolerance":   1e-10,           # 1e-10 absolute tolerance of the residual of the linear solver
      "solverType":          "lu",            # type of the linear solver
      "preconditionerType":  "none",          # type of the preconditioner
      "maxIterations":       1e4,                                         # maximum number of iterations in the linear solver
      "snesMaxFunctionEvaluations": 1e8,                                  # maximum number of function iterations
      "snesMaxIterations":   34,              # maximum number of iterations in the nonlinear solver
      "snesRelativeTolerance": 1e-2,         # relative tolerance of the nonlinear solver
      "snesAbsoluteTolerance": 1e-2,         # absolute tolerance of the nonlinear solver
      "snesLineSearchType": "l2",                                         # type of linesearch, possible values: "bt" "nleqerr" "basic" "l2" "cp" "ncglinear"
      "snesRebuildJacobianFrequency": 1,    # how often the jacobian should be recomputed, -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time the Jacobian is built etc. -2 means rebuild at next chance but then never again 
      "hypreOptions":        "",                                          # additional options for the hypre solvers could be given here
      "dumpFilename":        "",                                          # dump system matrix and right hand side after every solve
      "dumpFormat":          "matlab",                                    # default, ascii, matlab
    },
    "prestretchMechanicsSolver": {   # solver for the prestretch simulation (static mechanics problem)
      "relativeTolerance":   1e-5,          # 1e-10 relative tolerance of the linear solver
      "absoluteTolerance":   1e-10,         # 1e-10 absolute tolerance of the residual of the linear solver
      "solverType":          "lu",          # type of the linear solver
      "preconditionerType":  "none",        # type of the preconditioner
      "maxIterations":       1e4,           # maximum number of iterations in the linear solver
      "snesMaxFunctionEvaluations": 1e8,    # maximum number of function iterations
      "snesMaxIterations":   34,            # maximum number of iterations in the nonlinear solver
      "snesRelativeTolerance": 1e-10,       # relative tolerance of the nonlinear solver
      "snesAbsoluteTolerance": 1e-10,       # absolute tolerance of the nonlinear solver
      "snesLineSearchType": "l2",           # type of linesearch, possible values: "bt" "nleqerr" "basic" "l2" "cp" "ncglinear"
      "snesRebuildJacobianFrequency": 1,    # how often the jacobian should be recomputed, -1 indicates NEVER rebuild, 1 means rebuild every time the Jacobian is computed within a single nonlinear solve, 2 means every second time the Jacobian is built etc. -2 means rebuild at next chance but then never again 
      "hypreOptions":        "",            # additional options for the hypre solvers could be given here
      "dumpFilename":        "",            # dump system matrix and right hand side after every solve
      "dumpFormat":          "matlab",      # default, ascii, matlab
    },
    "potentialFlowSolver": {
      "relativeTolerance":  1e-10,
      "absoluteTolerance":  1e-10,         # 1e-10 absolute tolerance of the residual          
      "maxIterations":      1e5,
      "solverType":         variables.potential_flow_solver_type,
      "preconditionerType": variables.potential_flow_preconditioner_type,
      "dumpFormat":         "default",
      "dumpFilename":       "",
    },
    "multidomainLinearSolver": {
      "relativeTolerance":  variables.multidomain_relative_tolerance,
      "absoluteTolerance":  variables.multidomain_absolute_tolerance,         # 1e-10 absolute tolerance of the residual          
      "maxIterations":      variables.multidomain_max_iterations,             # maximum number of iterations
      "solverType":         variables.multidomain_solver_type,
      "preconditionerType": variables.multidomain_preconditioner_type,
      "hypreOptions":       "-pc_hypre_boomeramg_strong_threshold 0.7",       # additional options if a hypre preconditioner is selected
      "dumpFormat":         "matlab",
      "dumpFilename":       "",
          
      # gamg specific options:
      "gamgType":                         "agg",                          # one of agg, geo, or classical 
      "cycleType":                        "cycleW",                             # either cycleV or cycleW
      "nLevels":                          25,      
    },
    "multidomainAlternativeLinearSolver": {                                   # the alternative solver is used when the normal solver diverges
      "relativeTolerance":  variables.multidomain_relative_tolerance,
      "absoluteTolerance":  variables.multidomain_absolute_tolerance,         # 1e-10 absolute tolerance of the residual          
      "maxIterations":      variables.multidomain_alternative_solver_max_iterations,   # maximum number of iterations
      "solverType":         variables.multidomain_alternative_solver_type,
      "preconditionerType": variables.multidomain_alternative_preconditioner_type,
      "hypreOptions":       "",                                   # additional options if a hypre preconditioner is selected
      "dumpFormat":         "matlab",
      "dumpFilename":       "",
    },
    "mechanicsSolver": {   # solver for the dynamic mechanics problem
      "relativeTolerance":   variables.snes_relative_tolerance,           # 1e-10 relative tolerance of the linear solver
      "absoluteTolerance":   variables.snes_absolute_tolerance,           # 1e-10 absolute tolerance of the residual of the linear solver
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
  "MultipleCoupling": {
    "timeStepWidth":          variables.end_time,
    "logTimeStepWidthAsKey":  "dt_artifical_contraction",
    "durationLogKey":         "duration_artifical_contraction",
    "timeStepOutputInterval": 1,
    "endTime":                variables.end_time,
    "deferInitialization":    True,       # initialize nested solvers only right before computation of first timestep
    "connectedSlotsTerm1To2": None,       # connect lambda to slot 0 and gamma to slot 2
    "connectedSlotsTerm2To1": None,       # transfer nothing back
    
    # prescribed activation
    "Term1": {
      "PrescribedValues": {
        "meshName":               ["3Dmesh", "3DFatMesh"],      # reference to the multidomain mesh
        "numberTimeSteps":        1,             # number of timesteps to call the callback functions subsequently, this is usually 1 for prescribed values, because it is enough to set the reaction term only once per time step
        "timeStepOutputInterval": 20,            # if the time step should be written to console, a value > 10 produces no output
        "slotNames":              ["lambda", "gamma"],
        
        # a list of field variables that will get values assigned in every timestep, by the provided callback function
        "fieldVariables1": [
          {"name": "lambda", "callback": variables.set_lambda_values},
          {"name": "gamma", "callback": variables.set_gamma_values},
        ],
        "fieldVariables2":     [],
        "additionalArgument":  None,         # a custom argument to the fieldVariables callback functions, this will be passed on as the last argument
        
        "OutputWriter" : [
          {"format": "Paraview", "outputInterval": 1, "filename": "out/" + variables.scenario_name + "/0_prescribed_stress", "binary": True, "fixedFormat": False, "combineFiles": True, "fileNumbering": "incremental"}
        ]
      }
    },
    # artifical contraction with constant gamma
    "Term2": {  
      "MuscleContractionSolver": {
        "numberTimeSteps":              1,                         # only use 1 timestep per interval
        "timeStepOutputInterval":       100,                       # do not output time steps
        "Pmax":                         variables.pmax,            # maximum PK2 active stress
        "enableForceLengthRelation":    False,                     # if the factor f_l(λ_f) modeling the force-length relation (as in Heidlauf2013) should be multiplied. Set to false if this relation is already considered in the CellML model.
        "lambdaDotScalingFactor":       1.0,                       # scaling factor for the output of the lambda dot slot, i.e. the contraction velocity. Use this to scale the unit-less quantity to, e.g., micrometers per millisecond for the subcellular model.
        "slotNames":                    ["lambda", "ldot", "gamma", "T"],
        "OutputWriter" : [
          {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/"+variables.scenario_name+"/1_precontraction_mechanics_3D", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles": True, "fileNumbering": "incremental"},
        ],
        "mapGeometryToMeshes":          ["3Dmesh_elasticity_quadratic+3DFatMesh_elasticity_quadratic"],    # the mesh names of the meshes that will get the geometry transferred
        "reverseMappingOrder":          True,                      # if the mapping target->own mesh should be used instead of own->target mesh. This gives better results in some cases.
        "dynamic":                      False,                     # if the dynamic solid mechanics solver should be used, else it computes the quasi-static problem
        
        # the actual solid mechanics solver, this is either "DynamicHyperelasticitySolver" or "HyperelasticitySolver", depending on the value of "dynamic"
                                 
        "HyperelasticitySolver": {
          "durationLogKey":             "duration_precontraction",               # key to find duration of this solver in the log file
          
          "materialParameters":         variables.material_parameters,  # material parameters of the Mooney-Rivlin material
          "displacementsScalingFactor": 1.0,                       # scaling factor for displacements, only set to sth. other than 1 only to increase visual appearance for very small displacements
          "residualNormLogFilename":    "out/"+variables.scenario_name+"/1_log_precontraction_residual_norm.txt",   # log file where residual norm values of the nonlinear solver will be written
          "useAnalyticJacobian":        True,                      # whether to use the analytically computed jacobian matrix in the nonlinear solver (fast)
          "useNumericJacobian":         False,                     # whether to use the numerically computed jacobian matrix in the nonlinear solver (slow), only works with non-nested matrices, if both numeric and analytic are enable, it uses the analytic for the preconditioner and the numeric as normal jacobian
            
          "dumpDenseMatlabVariables":   False,                     # whether to have extra output of matlab vectors, x,r, jacobian matrix (very slow)
          # if useAnalyticJacobian,useNumericJacobian and dumpDenseMatlabVariables all all three true, the analytic and numeric jacobian matrices will get compared to see if there are programming errors for the analytic jacobian
          
          # mesh
          "inputMeshIsGlobal":          True,                     # boundary conditions and initial values are given as global numbers (every process has all information)
          "meshName":                   ["3Dmesh_elasticity_quadratic_precontraction", "3DFatMesh_elasticity_quadratic_precontraction"],       # name of the 3D mesh, it is defined under "Meshes" at the beginning of this config
          "fiberMeshNames":             [],                       # fiber meshes that will be used to determine the fiber direction, there are no fibers in multidomain, so this is empty
          "fiberDirection":             [0,0,1],                  # if fiberMeshNames is empty, directly set the constant fiber direction, in element coordinate system

          # solving
          "solverName":                 "precontractionMechanicsSolver",         # name of the nonlinear solver configuration, it is defined under "Solvers" at the beginning of this config
          #"loadFactors":                [0.5, 1.0],                # load factors for every timestep
          "loadFactors":                list(np.logspace(-1,0,3)),        # load factors
          #"loadFactors":                [],                        # no load factors
          "loadFactorGiveUpThreshold":  1e-5,                      # a threshold for the load factor, when to abort the solve of the current time step. The load factors are adjusted automatically if the nonlinear solver diverged. If the load factors get too small, it aborts the solve.
          "scaleInitialGuess":          False,                      # when load stepping is used, scale initial guess between load steps a and b by sqrt(a*b)/a. This potentially reduces the number of iterations per load step (but not always).
          "nNonlinearSolveCalls":       1,                         # how often the nonlinear solve should be repeated
          
          # boundary and initial conditions
          "dirichletBoundaryConditions": variables.precontraction_elasticity_dirichlet_bc,   # the initial Dirichlet boundary conditions that define values for displacements u and velocity v
          "neumannBoundaryConditions":   variables.precontraction_elasticity_neumann_bc,     # Neumann boundary conditions that define traction forces on surfaces of elements
          "divideNeumannBoundaryConditionValuesByTotalArea": True,            # if the given Neumann boundary condition values under "neumannBoundaryConditions" are total forces instead of surface loads and therefore should be scaled by the surface area of all elements where Neumann BC are applied
          "updateDirichletBoundaryConditionsFunction": None,                  # function that updates the dirichlet BCs while the simulation is running
          "updateDirichletBoundaryConditionsFunctionCallInterval": 1,         # every which step the update function should be called, 1 means every time step
          
          "initialValuesDisplacements":  [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the displacements, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
          "initialValuesVelocities":     [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the velocities, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
          "extrapolateInitialGuess":     True,                                # if the initial values for the dynamic nonlinear problem should be computed by extrapolating the previous displacements and velocities
          "constantBodyForce":           variables.precontraction_constant_body_force,       # a constant force that acts on the whole body, e.g. for gravity
          
          "dirichletOutputFilename":    "out/"+variables.scenario_name+"/1_precontraction_dirichlet_boundary_conditions",     # output filename for the dirichlet boundary conditions, set to "" to have no output
          
          # define which file formats should be written
          # 1. main output writer that writes output files using the quadratic elements function space. Writes displacements, velocities and PK2 stresses.
          "OutputWriter" : [
            
            # Paraview files
            {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/1_precontraction_u", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
          ],
          # 2. additional output writer that writes also the hydrostatic pressure
          "pressure": {   # output files for pressure function space (linear elements), contains pressure values, as well as displacements and velocities
            "OutputWriter" : [
              {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/1_precontraction_p", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
            ]
          },
          # 4. output writer for debugging, outputs files after each load increment, the geometry is not changed but u and v are written
          "LoadIncrements": {   
            "OutputWriter" : [
              {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/1_precontraction_load_increments", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
            ]
          }
        }
      }
    },
    # prestretch simulation
    "Term3": {
      "HyperelasticitySolver": {
        "durationLogKey":             "duration_prestretch",               # key to find duration of this solver in the log file
        
        "materialParameters":         variables.material_parameters,  # material parameters of the Mooney-Rivlin material
        "displacementsScalingFactor": 1.0,                       # scaling factor for displacements, only set to sth. other than 1 only to increase visual appearance for very small displacements
        "residualNormLogFilename":    "out/"+variables.scenario_name+"/2_log_prestretch_residual_norm.txt",   # log file where residual norm values of the nonlinear solver will be written
        "useAnalyticJacobian":        True,                      # whether to use the analytically computed jacobian matrix in the nonlinear solver (fast)
        "useNumericJacobian":         False,                     # whether to use the numerically computed jacobian matrix in the nonlinear solver (slow), only works with non-nested matrices, if both numeric and analytic are enable, it uses the analytic for the preconditioner and the numeric as normal jacobian
        "slotNames":                  ["m_ux", "m_uy", "m_uz"],  # slot names of the data connector slots, there are three slots, namely the displacement components ux, uy, uz
          
        "dumpDenseMatlabVariables":   False,                     # whether to have extra output of matlab vectors, x,r, jacobian matrix (very slow)
        # if useAnalyticJacobian,useNumericJacobian and dumpDenseMatlabVariables all all three true, the analytic and numeric jacobian matrices will get compared to see if there are programming errors for the analytic jacobian
        
        # mesh
        "inputMeshIsGlobal":          True,                     # boundary conditions and initial values are given as global numbers (every process has all information)
        "meshName":                   ["3Dmesh_elasticity_quadratic", "3DFatMesh_elasticity_quadratic"],       # name of the 3D mesh, it is defined under "Meshes" at the beginning of this config
        "fiberMeshNames":             [],                       # fiber meshes that will be used to determine the fiber direction, there are no fibers in multidomain, so this is empty
        "fiberDirection":             [0,0,1],                  # if fiberMeshNames is empty, directly set the constant fiber direction, in element coordinate system

        # solving
        "solverName":                 "prestretchMechanicsSolver",         # name of the nonlinear solver configuration, it is defined under "Solvers" at the beginning of this config
        #"loadFactors":                [0.5, 1.0],                # load factors for every timestep
        "loadFactors":                list(np.logspace(-1,0,3)),        # load factors
        #"loadFactors":                [],                        # no load factors
        "loadFactorGiveUpThreshold":  1e-5,                      # a threshold for the load factor, when to abort the solve of the current time step. The load factors are adjusted automatically if the nonlinear solver diverged. If the load factors get too small, it aborts the solve.
        "scaleInitialGuess":          False,                      # when load stepping is used, scale initial guess between load steps a and b by sqrt(a*b)/a. This potentially reduces the number of iterations per load step (but not always).
        "nNonlinearSolveCalls":       1,                         # how often the nonlinear solve should be repeated
        
        # boundary and initial conditions
        "dirichletBoundaryConditions": variables.prestretch_elasticity_dirichlet_bc,   # the initial Dirichlet boundary conditions that define values for displacements u and velocity v
        "neumannBoundaryConditions":   variables.prestretch_elasticity_neumann_bc,     # Neumann boundary conditions that define traction forces on surfaces of elements
        "divideNeumannBoundaryConditionValuesByTotalArea": True,            # if the given Neumann boundary condition values under "neumannBoundaryConditions" are total forces instead of surface loads and therefore should be scaled by the surface area of all elements where Neumann BC are applied
        "updateDirichletBoundaryConditionsFunction": None,                  # function that updates the dirichlet BCs while the simulation is running
        "updateDirichletBoundaryConditionsFunctionCallInterval": 1,         # every which step the update function should be called, 1 means every time step
        
        "initialValuesDisplacements":  [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the displacements, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
        "initialValuesVelocities":     [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the velocities, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
        "extrapolateInitialGuess":     True,                                # if the initial values for the dynamic nonlinear problem should be computed by extrapolating the previous displacements and velocities
        "constantBodyForce":           variables.prestretch_constant_body_force,       # a constant force that acts on the whole body, e.g. for gravity
        
        "dirichletOutputFilename":    "out/"+variables.scenario_name+"/2_prestretch_dirichlet_boundary_conditions",     # output filename for the dirichlet boundary conditions, set to "" to have no output
        
        # define which file formats should be written
        # 1. main output writer that writes output files using the quadratic elements function space. Writes displacements, velocities and PK2 stresses.
        "OutputWriter" : [
          
          # Paraview files
          {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/2_prestretch_u", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
        ],
        # 2. additional output writer that writes also the hydrostatic pressure
        "pressure": {   # output files for pressure function space (linear elements), contains pressure values, as well as displacements and velocities
          "OutputWriter" : [
            {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/2_prestretch_p", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
          ]
        },
        # 4. output writer for debugging, outputs files after each load increment, the geometry is not changed but u and v are written
        "LoadIncrements": {   
          "OutputWriter" : [
            {"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/2_prestretch_load_increments", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
          ]
        },
      }
    },
    # actual Multidomain simulation
    "Term4": {
      "Coupling": {
        "description":            "actual multidomain simulation",    # description that will be shown in solver structure visualization
        "timeStepWidth":          variables.dt_elasticity,
        "logTimeStepWidthAsKey":  "dt_elasticity",
        "durationLogKey":         "duration_total",
        "timeStepOutputInterval": 1,
        "endTime":                variables.end_time,
        "connectedSlotsTerm1To2": None,       # data transfer is configured using global option "connectedSlots"
        "connectedSlotsTerm2To1": None,       # transfer nothing back
        "Term1": {        # multidomain
          "StrangSplitting": {
            "description":            "multidomain",    # description that will be shown in solver structure visualization
            "timeStepWidth":          variables.dt_splitting,
            "logTimeStepWidthAsKey":  "dt_splitting",
            "durationLogKey":         "duration_total",
            "timeStepOutputInterval": 100,
            "endTime":                variables.end_time,
            "connectedSlotsTerm1To2": [0],          # CellML V_mk (0) <=> Multidomain V_mk^(i) (0)
            "connectedSlotsTerm2To1": [None, 0],    # Multidomain V_mk^(i+1) (1) -> CellML V_mk (0)

            "Term1": {      # CellML
              "MultipleInstances": {
                "nInstances": variables.n_compartments,  
                "instances": [        # settings for each motor unit, `i` is the index of the motor unit
                {
                  "ranks": list(range(n_ranks)),
                  "Heun" : {
                    "description":                  "0D of multidomain",    # description that will be shown in solver structure visualization
                    "timeStepWidth":                variables.dt_0D,  # 5e-5
                    "logTimeStepWidthAsKey":        "dt_0D",
                    "durationLogKey":               "duration_0D",
                    "initialValues":                [],
                    "timeStepOutputInterval":       1e4,
                    "inputMeshIsGlobal":            True,
                    "dirichletBoundaryConditions":  {},
                    "dirichletOutputFilename":      None,             # output filename for the dirichlet boundary conditions, set to "" to have no output
                    "checkForNanInf":               True,             # check if the solution vector contains nan or +/-inf values, if yes, an error is printed. This is a time-consuming check.
                    "nAdditionalFieldVariables":    0,
                    "additionalSlotNames":          [],
                        
                    "CellML" : {
                      "modelFilename":                          variables.cellml_file,                          # input C++ source file or cellml XML file
                      "initializeStatesToEquilibrium":          False,                                          # if the equilibrium values of the states should be computed before the simulation starts
                      "initializeStatesToEquilibriumTimestepWidth": 1e-4,                                       # if initializeStatesToEquilibrium is enable, the timestep width to use to solve the equilibrium equation
                      
                      # optimization parameters
                      "optimizationType":                       "vc",                                           # "vc", "simd", "openmp" type of generated optimizated source file
                      "approximateExponentialFunction":         True,                                           # if optimizationType is "vc", whether the exponential function exp(x) should be approximate by (1+x/n)^n with n=1024
                      "compilerFlags":                          "-fPIC -O3 -march=native -shared ",             # compiler flags used to compile the optimized model code
                      "maximumNumberOfThreads":                 0,                                              # if optimizationType is "openmp", the maximum number of threads to use. Default value 0 means no restriction.
                      
                      # stimulation callbacks
                      "setSpecificStatesFunction":              set_specific_states,                                             # callback function that sets states like Vm, activation can be implemented by using this method and directly setting Vm values, or by using setParameters/setSpecificParameters
                      #"setSpecificStatesCallInterval":         2*int(1./variables.stimulation_frequency/variables.dt_0D),       # set_specific_states should be called variables.stimulation_frequency times per ms, the factor 2 is needed because every Heun step includes two calls to rhs
                      "setSpecificStatesCallInterval":          0,                                                               # 0 means disabled
                      "setSpecificStatesCallFrequency":         variables.get_specific_states_call_frequency(compartment_no),   # set_specific_states should be called variables.stimulation_frequency times per ms
                      "setSpecificStatesFrequencyJitter":       variables.get_specific_states_frequency_jitter(compartment_no), # random value to add or substract to setSpecificStatesCallFrequency every stimulation, this is to add random jitter to the frequency
                      "setSpecificStatesRepeatAfterFirstCall":  0.01,                                                            # [ms] simulation time span for which the setSpecificStates callback will be called after a call was triggered
                      "setSpecificStatesCallEnableBegin":       variables.get_specific_states_call_enable_begin(compartment_no),# [ms] first time when to call setSpecificStates
                      "additionalArgument":                     compartment_no,
                      
                      "mappings":                               variables.mappings,                             # mappings between parameters and algebraics/constants and between outputConnectorSlots and states, algebraics or parameters, they are defined in helper.py
                      "parametersInitialValues":                variables.parameters_initial_values,            #[0.0, 1.0],      # initial values for the parameters: I_Stim, l_hs
                      
                      "meshName":                               "3Dmesh",                                       # use the linear mesh, it was partitioned by the helper.py script which called opendihu/scripts/create_partitioned_meshes_for_settings.py
                      "stimulationLogFilename":                 "out/" + variables.scenario_name + "/stimulation.log",
      
                      # output writer for states, algebraics and parameters                
                      "OutputWriter" : [
                        {"format": "Paraview", "outputInterval": (int)(1./variables.dt_0D*variables.output_timestep_multidomain), "filename": "out/" + variables.scenario_name + "/3_0D_all", "binary": True, "fixedFormat": False, "combineFiles": True, "fileNumbering": "incremental"}
                      ] if variables.states_output else []
                    },

                    # output writer for states
                    "OutputWriter" : [
                      {"format": "Paraview", "outputInterval": (int)(1./variables.dt_multidomain*variables.output_timestep_multidomain), "filename": "out/" + variables.scenario_name + "/3_0D_states", "binary": True, "fixedFormat": False, "combineFiles": True, "fileNumbering": "incremental"}
                    ] if variables.states_output else []
                  }
                } for compartment_no in range(variables.n_compartments)]
              },
            },
            "Term2": {     # Diffusion, i.e. Multidomain
              "MultidomainSolver": multidomain_solver,
              "OutputSurface": {        # version for fibers_emg_2d_output
                "OutputWriter": [
                  {"format": "Paraview", "outputInterval": (int)(1./variables.dt_multidomain*variables.output_timestep_multidomain), "filename": "out/"+variables.scenario_name+"/3_surface", "binary": True, "fixedFormat": False, "combineFiles": True, "fileNumbering": "incremental"},
                ],
                "face":                     ["1+"],              # which faces of the 3D mesh should be written into the 2D mesh
                "samplingPoints":           [],  #variables.hdemg_electrode_positions,    # the electrode positions, they are created in the helper.py script
                "updatePointPositions":     False,               # the electrode points should be initialize in every timestep (set to False for the static case). This makes a difference if the muscle contracts, then True=fixed electrodes, False=electrodes moving with muscle.
                "filename":                 "out/{}/electrodes.csv".format(variables.scenario_name),
                "enableCsvFile":            True,                # if the values at the sampling points should be written to csv files
                "enableVtpFile":            False,               # if the values at the sampling points should be written to vtp files
                "enableGeometryInCsvFile":  False,               # if the csv output file should contain geometry of the electrodes in every time step. This increases the file size and only makes sense if the geometry changed throughout time, i.e. when computing with contraction
                "xiTolerance":              0.3,                 # tolerance for element-local coordinates xi, for finding electrode positions inside the elements. Increase or decrease this numbers if not all electrode points are found.

                "MultidomainSolver": multidomain_solver,
              }
            }
          }
        },
        "Term2": {        # solid mechanics
          "MuscleContractionSolver": {
            "numberTimeSteps":              1,                         # only use 1 timestep per interval
            "timeStepOutputInterval":       100,                       # do not output time steps
            "Pmax":                         variables.pmax,            # maximum PK2 active stress
            "enableForceLengthRelation":    True,                      # if the factor f_l(λ_f) modeling the force-length relation (as in Heidlauf2013) should be multiplied. Set to false if this relation is already considered in the CellML model.
            "lambdaDotScalingFactor":       1.0,                       # scaling factor for the output of the lambda dot slot, i.e. the contraction velocity. Use this to scale the unit-less quantity to, e.g., micrometers per millisecond for the subcellular model.
            "slotNames":                    ["m_lda", "m_ldot", "m_g_in", "m_T", "m_ux", "m_uy", "m_uz"],  # slot names of the data connector slots: lambda, lambdaDot, gamma, traction
            "OutputWriter" : [
              {"format": "Paraview", "outputInterval": int(1./variables.dt_elasticity*variables.output_timestep_elasticity), "filename": "out/" + variables.scenario_name + "/4_mechanics_3D", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles": True, "fileNumbering": "incremental"},
            ],
            "mapGeometryToMeshes":          ["3Dmesh","3DFatMesh"],    # the mesh names of the meshes that will get the geometry transferred
            "reverseMappingOrder":          True,                      # if the mapping target->own mesh should be used instead of own->target mesh. This gives better results in some cases.
            "dynamic":                      True,                      # if the dynamic solid mechanics solver should be used, else it computes the quasi-static problem
            
            # the actual solid mechanics solver, this is either "DynamicHyperelasticitySolver" or "HyperelasticitySolver", depending on the value of "dynamic"
            "DynamicHyperelasticitySolver": {
              "timeStepWidth":              variables.dt_elasticity,           # time step width 
              "durationLogKey":             "duration_mechanics",               # key to find duration of this solver in the log file
              "timeStepOutputInterval":     1,                         # how often the current time step should be printed to console
              
              "materialParameters":         variables.material_parameters,  # material parameters of the Mooney-Rivlin material
              "density":                    variables.rho,             # density of the material
              "displacementsScalingFactor": 1.0,                       # scaling factor for displacements, only set to sth. other than 1 only to increase visual appearance for very small displacements
              "residualNormLogFilename":    "out/"+variables.scenario_name+"/4_multidomain_log_residual_norm.txt",   # log file where residual norm values of the nonlinear solver will be written
              "useAnalyticJacobian":        True,                      # whether to use the analytically computed jacobian matrix in the nonlinear solver (fast)
              "useNumericJacobian":         False,                     # whether to use the numerically computed jacobian matrix in the nonlinear solver (slow), only works with non-nested matrices, if both numeric and analytic are enable, it uses the analytic for the preconditioner and the numeric as normal jacobian
                
              "dumpDenseMatlabVariables":   False,                     # whether to have extra output of matlab vectors, x,r, jacobian matrix (very slow)
              # if useAnalyticJacobian,useNumericJacobian and dumpDenseMatlabVariables all all three true, the analytic and numeric jacobian matrices will get compared to see if there are programming errors for the analytic jacobian
              
              # mesh
              "inputMeshIsGlobal":          True,                     # boundary conditions and initial values are given as global numbers (every process has all information)
              "meshName":                   ["3Dmesh_elasticity_quadratic", "3DFatMesh_elasticity_quadratic"],       # name of the 3D mesh, it is defined under "Meshes" at the beginning of this config
              "fiberMeshNames":             [],                       # fiber meshes that will be used to determine the fiber direction, there are no fibers in multidomain, so this is empty
              "fiberDirection":             [0,0,1],                  # if fiberMeshNames is empty, directly set the constant fiber direction, in element coordinate system
        
              # solving
              "solverName":                 "mechanicsSolver",         # name of the nonlinear solver configuration, it is defined under "Solvers" at the beginning of this config
              #"loadFactors":                [0.5, 1.0],                # load factors for every timestep
              "loadFactors":                [],                        # no load factors, solve problem directly
              "loadFactorGiveUpThreshold":  0.5,                       # a threshold for the load factor, when to abort the solve of the current time step. The load factors are adjusted automatically if the nonlinear solver diverged. If the load factors get too small, it aborts the solve.
              "scaleInitialGuess":          False,                     # when load stepping is used, scale initial guess between load steps a and b by sqrt(a*b)/a. This potentially reduces the number of iterations per load step (but not always).
              "nNonlinearSolveCalls":       1,                         # how often the nonlinear solve should be repeated
              
              # boundary and initial conditions
              "dirichletBoundaryConditions": variables.multidomain_elasticity_dirichlet_bc,   # the initial Dirichlet boundary conditions that define values for displacements u and velocity v
              "neumannBoundaryConditions":   variables.multidomain_elasticity_neumann_bc,     # Neumann boundary conditions that define traction forces on surfaces of elements
              "divideNeumannBoundaryConditionValuesByTotalArea": True,            # if the given Neumann boundary condition values under "neumannBoundaryConditions" are total forces instead of surface loads and therefore should be scaled by the surface area of all elements where Neumann BC are applied
              "updateDirichletBoundaryConditionsFunction": None,                  # function that updates the dirichlet BCs while the simulation is running
              "updateDirichletBoundaryConditionsFunctionCallInterval": 1,         # every which step the update function should be called, 1 means every time step
              
              "initialValuesDisplacements":  [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the displacements, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
              "initialValuesVelocities":     [[0.0,0.0,0.0] for _ in range(variables.n_points_global_composite_mesh)],     # the initial values for the velocities, vector of values for every node [[node1-x,y,z], [node2-x,y,z], ...]
              "extrapolateInitialGuess":     True,                                # if the initial values for the dynamic nonlinear problem should be computed by extrapolating the previous displacements and velocities
              "constantBodyForce":           variables.multidomain_constant_body_force,       # a constant force that acts on the whole body, e.g. for gravity
              
              "dirichletOutputFilename":     "out/"+variables.scenario_name+"/4_multidomain_dirichlet_boundary_conditions",     # output filename for the dirichlet boundary conditions, set to "" to have no output
              "totalForceLogFilename":       "out/"+variables.scenario_name+"/4_multidomain_tendon_force.csv",              # filename of a log file that will contain the total (bearing) forces and moments at the top and bottom of the volume
              "totalForceLogOutputInterval":       10,                                  # output interval when to write the totalForceLog file
              "totalForceBottomElementNosGlobal":  [j*nx + i for j in range(ny) for i in range(nx)],                  # global element nos of the bottom elements used to compute the total forces in the log file totalForceLogFilename
              "totalForceTopElementNosGlobal":     [(nz-1)*ny*nx + j*nx + i for j in range(ny) for i in range(nx)],   # global element nos of the top elements used to compute the total forces in the log file totalForceTopElementsGlobal

              # define which file formats should be written
              # 1. main output writer that writes output files using the quadratic elements function space. Writes displacements, velocities and PK2 stresses.
              "OutputWriter" : [
                
                # Paraview files
                #{"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/4_displacements", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                
                # Python callback function "postprocess"
                #{"format": "PythonCallback", "outputInterval": 1, "callback": postprocess, "onlyNodalValues":True, "filename": ""},
              ],
              # 2. additional output writer that writes also the hydrostatic pressure
              "pressure": {   # output files for pressure function space (linear elements), contains pressure values, as well as displacements and velocities
                "OutputWriter" : [
                  #{"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/4_pressure", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ]
              },
              # 3. additional output writer that writes virtual work terms
              "dynamic": {    # output of the dynamic solver, has additional virtual work values 
                "OutputWriter" : [   # output files for displacements function space (quadratic elements)
                  #{"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/dynamic", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                  #{"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/4_virtual_work", "binary": True, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ],
              },
              # 4. output writer for debugging, outputs files after each load increment, the geometry is not changed but u and v are written
              "LoadIncrements": {   
                "OutputWriter" : [
                  #{"format": "Paraview", "outputInterval": 1, "filename": "out/"+variables.scenario_name+"/4_load_increments", "binary": False, "fixedFormat": False, "onlyNodalValues":True, "combineFiles":True, "fileNumbering": "incremental"},
                ]
              },
            }
          }
        }
      }
    }
  }
}

# stop timer and calculate how long parsing lasted
if rank_no == 0:
  t_stop_script = timeit.default_timer()
  print("Python config parsed in {:.1f}s.".format(t_stop_script - t_start_script))

