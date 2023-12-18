from .Edited_llesolver import LLEsolver #Just need to change the Julia file that the thing uses
from .DintFuncs import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import pickle as pkl
import pandas as pd
import sys
import os
pio.renderers.default = "notebook"
pio.templates.default = "seaborn"


class pyLLEHelper(LLEsolver):
	'''
	Class that makes using pyLLE more streamlined for soliton perturbations. All simulations will
	Return a pyLLEHelper object containing the relavant data/simulation. Various plotting functions
	included.

	To be continously edited for the time being.
	'''
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bin = self.getBin()
		if not "Manual_Detuning" in self._sim.keys():
			self._sim["Manual_Detuning"] = 0
		if not "Manual_Pin" in self._sim.keys():
			self._sim["Manual_Pin"] = 0

		# These are here for ease on typing. May need to change the [] part of Pin
		self.Tscan = self._sim["Tscan"] 
		self.Pin = self._sim["Pin"][0]
		self.N_mu = self._sim["mu_sim"][1]
		self.num_probe = self._sim["num_probe"]

		if self._sim["domega_end"] == self._sim["domega_init"]:
			self.detuning = self._sim["domega_init"]
		else:
			self.detuning = None

	def getBin(self):
		'''
		Gets the path to Julia application. Needed to run SolveTemporal().
		'''
		if sys.platform == 'darwin':
			julia = '/Applications/Julia-1.9.app/Contents/Resources/julia/bin/julia'
		if sys.platform == 'linux2':
			julia = 'julia'
		if sys.platform == 'win32':
			#EDIT THIS SO IT WORKS FOR MARK
			julia = os.path.expanduser('~') + '\\AppData\\Local\\Programs\\Julia-1.9.3\\bin\\julia.exe'
		return julia

	# -- These two functions likely unnecessary, yet keeping them in for now
	#-----------------------------------------------------------------
	def addManualDispersion(self,Manual_Detuning=0):
		if Manual_Pin == 0 or len(Manual_Detuning) == self.sim.Tscan:
			self._sim.update({"Manual_Pin":Manual_Detuning})
		else:
			print("Inputted Manual Detuning of the wrong size")
			print(f"Given array of size {len(Manual_Detuning)}, expecting an array of size {self.sim.Tscan}")

	def addManualPin(self,Manual_Pin=0):
		if Manual_Pin == 0 or len(Manual_Pin) == self.sim.Tscan:
			self._sim.update({"Manual_Pin":Manual_Pin})
		else:
			print("Inputted Manual Pin of the wrong size")
			print(f"Given array of size {len(Manual_Pin)}, expecting array of size {self.sim.Tscan}")

	#-----------------------------------------------------------------

	def calcNewD1(self):
		'''
		Calculates new/adjusted D1 from a drifting soliton. Necessary for "freezing" the soliton

		Returns:
			newD1 - adjusted D1 value which should "freeze" the soliton when set as D1_manual.
		'''
		cyclesPerStep = self.Tscan/self.num_probe
		slope, _ = self.calcThetaSlope()
		Trt0 = 2*np.pi/self._sim['D1']
		frep = 1/Trt0 * (1-slope/(cyclesPerStep))
		newD1 = 2*np.pi*frep
		return newD1

	def calcThetaSlope(self,convolutionSpread=30):
		'''
		Extrapolates the slope of the soliton position vs time graph. Slope is in round trips (cycles) traveled per time step.
		'''
		thetas = np.linspace(0,2*np.pi, self.sol.Acav.shape[0])
		cycles_per_time = np.unwrap(thetas[np.argmax(np.abs(self.sol.Acav)**2,0)])/(2*np.pi)
		slope = np.average(np.gradient(cycles_per_time))
		# May need to refine method for calculating slope, leaving this convolution stuff in for now
		#slope = (np.average(np.convolve(np.gradient(cycles_per_time),np.ones(convolutionSpread)/convolutionSpread)))
		return slope, cycles_per_time

	def getDKSdata(self,detuningIdx):
		'''
		Retrieves Soliton data from solver object. To be used as an initialized state.

		Params:
			detuningIdx - detuning index defining location of wanted initial soliton state

		Returns:
			_Acav - array containing intra-cavity field data at given detuning. (u_probe is what it wants)
			δω - detuning in rad/s
		'''
		_Acav = self._sol['u_probe'][:,detuningIdx]
		δω = self.sol.δfreq[detuningIdx] * 2*np.pi
		return _Acav, δω

	def customAnalyze(self,wg_name = 'TE_18'):
		'''
		Analyze function that creates Dint, mu_sim_center, f_center, ind_pump, and DKS_init. Needed for Julia code to 
		run properly. Dint is a combination of measured data and simulated data
		'''
		self.calcDint(wg_name)
		self._sim['mu_sim_center'] = self._sim['mu_sim']
		self._sim['f_center'] = self._sim['f_pmp']
		self._sim['ind_pmp'] = [0]
		if not "DKS_init" in self._sim.keys():
			self._sim["DKS_init"] = np.zeros(self._sim['Dint'].size)

	def calcDint(self,wg_name='TE_18'):
		df = pd.read_csv('./SiN_Dispersion_Rect_Wg.csv')
		df1 = pd.read_csv(self._res['dispfile'])
		simAngFreqs = np.array([x for x in df[wg_name+' w'].values if str(x) != 'nan'])
		simD = np.array([x for x in df[wg_name+' D'].values if str(x) != 'nan'])
		measResFreqs = np.sort(df1['Resonance Frequency'].values)*1e12

		measDint, D1, pmp_idx, mu, mu_corr = calcMeasDint(measResFreqs,self._sim['f_pmp'][0],D1_manual = self._sim['D1_manual'])
		if not self._sim['D1_manual']:
			measDint, D1, mu_corr = updateDint(measDint, D1, measResFreqs, pmp_idx, mu)
		simDint, simResAngFreqs = calcSimDint(simAngFreqs, simD, D1, measResFreqs[pmp_idx]*2*np.pi,self.N_mu) #Setting f_pmp to the freq closest to given f_pmp in f list.
		connDint = connectDint(simDint, measDint, mu_corr, self.N_mu, pmp_idx)
		plt.scatter(simResAngFreqs/2/np.pi,connDint)
		#plt.xlim(1.8e14,2.1e14)
		#plt.ylim(-50e10,50e10)
		print("man D1",self._sim['D1_manual'],'calc D1', D1, D1==self._sim['D1_manual'])
		self._res["ng"] = const.c/(np.gradient(simResAngFreqs/2/np.pi * 2*np.pi*self._res['R']))[pmp_idx] #copy paste ng calc. Need for stuff to not complain when running
		self._sim["Dint"] = connDint 
		self._sim["D1"] = D1 #If manual D1, prob dont need to call updateDint. will prob change D1 value
		self._sim["FSR"] = [D1/2/np.pi]
		self._sim["FSR_center"] = D1/2/np.pi
		self._sim["D1_center"] = D1
		self._sim["f_pmp"] = np.array([measResFreqs[pmp_idx]])
		self.simDint = simDint
		self.measDint = measDint
		self.connDint = connDint
		self.simAngFreqs = simResAngFreqs
		self.measResFreqs = measResFreqs


	def freezeSolitonSim(self,detuningIdx=-1,simDrifting=False,saveSolvers=False,custom_analyze=False):
		'''
		Freezes a soliton. Can either start with a solver object containing detuning sweep data OR a
		solver object with drifting soliton data.

		Params:
			deturningIdx - detuning index corresponding to wanted initial soliton state.
			saveSolvers - True if you want to save drifting and frozen solvers. Will save as .pkl in current dir.
			simDrifting - True if drifting sim is needed, False if not.
			plotDrifting - True to plot drifting plot, False to not plot.
			plotFrozen - True to plot frozen soliton plot, Flase to not plot.

		Returns:
			frozenSolver - pyLLEHelper object for frozen soliton.
			driftingSolver - pyLLEHelper object for drifting soliton.
		'''
		if simDrifting:
			if detuningIdx == -1:
				raise Exception("Forgot to input detuning index.")
			DKSSolver = self.DKSSim(detuningIdx, saveSolvers,custom_analyze)
		else:
			DKSSolver = self

		DKSinit, δω = DKSSolver._sim['DKS_init'],DKSSolver._sim['domega_end']
		newD1 = DKSSolver.calcNewD1()
		freezeSim = DKSSolver.update_sim(DKSinit,δω,newD1)

		print("Running Simulation from Initialized State with adjusted D1")
		frozenSolver = pyLLEHelper(sim=freezeSim, res=self._res,debug=False)
		frozenSolver.runSim(saveSolvers,custom_analyze,"_Frozen_DKS_Sim")

		return frozenSolver,DKSSolver

	def update_sim(self,DKSinit,δω,newD1=None):
		'''
		Adds the necessary data to the '_sim' dictionary. Needed as parameter for a LLEsolver object.

		Params:
			DKSinit - initial intracavity field data
			δω - detuning in rad/s of initial state
			newD1 - adjusted D1 value to "freeze" soliton

		Returns:
			_sim - _sim dictionary with updated values. Ready for a DKS simulation.
		'''
		_sim = self._sim.copy()
		_sim.update({'δω_end':δω,'δω_init':δω,'DKS_init':DKSinit,'D1_manual':newD1})
		return _sim

	# Can make things less confusing by having this function deal with all the saving, just pass the wanted file name
	# runSim() is used in every other function so might as well
	def runSim(self,saveSolver=False,custom_analyze=False,fname='_Untitled_Sim',path="./"):
		'''
		Runs the simulation. Cranks out the sol object and all of the fields.

		Params:
			saveSolver - True if want to save pyLLEHelper, False if not. Will save in current dir as a .pkl
			fname - string to name the saved pyLLEHelper. Will be concatenated to the end of the data file name.
		'''
		if custom_analyze:
			self.customAnalyze()
		else:
			_ = self.Analyze(plot=False) #Adjust this function to plot Dint if wanted
		self.Setup(verbose=False)
		self.SolveTemporal(bin=self.bin)
		self.RetrieveData()
		if saveSolver:
			self.SaveResults(self._res['dispfile'][:-4]+fname,path)

	def detuningSweepSim(self,saveSolver=False,custom_analyze=True):
		'''
		Runs detuning sweep simulation. This is to be used when the solver only has mode/resonance freq data.
		It will produce the sol object and all of the calulated fields.
		'''
		self.runSim(saveSolver,custom_analyze,"_Detuning_Sweep_Sim")

	def DKSSim(self,detuningIdx,saveSolver=False,custom_analyze=False):
		DKSinit, δω = self.getDKSdata(detuningIdx)
		DKSSim = self.update_sim(DKSinit,δω)

		print("Running Simulation from Initialized State")
		DKSSolver = pyLLEHelper(sim=DKSSim, res=self._res,debug=False)
		DKSSolver.runSim(saveSolver,custom_analyze,"_DKS_Sim")
		return DKSSolver

	def pinPerturbationSim(self, Manual_Pin,saveSolver=False,custom_analyze=False):
		'''
		Runs Pin Perturbation simulation. This is to be run on a frozen soliton to see the effects of perturbing the input
		pump power.

		Params:
			Manual_Pin - list/array of length Tsan. Holds the perturbed input pump power for each round trip.
			saveSolver - True if want to save pyLLEHelper as a .pkl.

		Returns:
			pinSolver - pyLLEHelper object with pin perturbed data.
		'''
		if len(Manual_Pin) != self.sim.Tscan:
			raise Exception(f"Given array of size {len(Manual_Pin)}, expecting array of size {self.sim.Tscan}")
		
		perturb_sim = self._sim.copy()
		perturb_sim.update({"Manual_Pin":Manual_Pin})
		pinSolver = pyLLEHelper(sim=perturb_sim,res=self._res,debug=False)
		pinSolver.runSim(saveSolver,custom_analyze,'_Pin_Perturbation_Sim')

		return pinSolver

	def detuningPerturbationSim(self,Manual_Detuning,saveSolver=False,custom_analyze=False):
		'''
		Runs Detuning Perturbation simulation. Needs to be run off a frozen soliton to get a better
		idea of the perturbing effects.

		Params:
			Manual_Detuning - list/array of length Tsan. Holds the perturbed detuning values for each round trip.
			saveSolver - True if want to save pyLLEHelper as a .pkl.

		Returns:
			pinSolver - pyLLEHelper object with detuning perturbed data.
		'''
		if len(Manual_Detuning) != self.sim.Tscan:
			raise Exception(f"Given array of size {len(Manual_Detuning)}, expecting array of size {self.sim.Tscan}")

		perturb_sim = self._sim.copy()
		perturb_sim.update({"Manual_Detuning":Manual_Detuning})
		detuningSolver = pyLLEHelper(sim=perturb_sim,res=self._res,debug=False)
		detuningSolver.runSim(saveSolver,custom_analyze,"_Detuning_Perturbation_Sim")

		return detuningSolver

	def detuningAndPinPerturbationSim(self,Manual_Detuning,Manual_Pin,saveSolver=False,custom_analyze=False):
		'''
		Runs Detuning and Pin Perturbation simulation at the same time. Needs to be run off a frozen soliton to get a better
		idea of the perturbing effects.

		Params:
			Manual_Detuning - list/array of length Tsan. Holds the perturbed detuning values for each round trip.
			Manual_Pin -  list/array of length Tsan. Holds the perturbed input pump power for each round trip.
			saveSolver - True if want to save pyLLEHelper as a .pkl.

		Returns:
			doublyPerturbedSolver - pyLLEHelper object with detuning and Pin perturbed data.
		'''
		if len(Manual_Detuning) != self.Tscan or len(Manual_Pin) != self.Tscan:
			raise Exception(f"Given Manual_Detuning array of size {len(Manual_Detuning)}, expecting array of size {self.Tscan}\n"+
							f"Given Manual_Pin array of size {len(Manual_Pin)}, expecting array of size {self.Tscan}")
		perturb_sim = self._sim.copy()
		perturb_sim.update({"Manual_Detuning":Manual_Detuning,"Manual_Pin":Manual_Pin})
		doublyPerturbSolver = pyLLEHelper(sim=perturb_sim,res=self._res,debug=False)
		doublyPerturbSolver.runSim(saveSolver,custom_analyze,"_Doubly_Perturbed_Sim")

		return doublyPerturbSolver

	def plotMaxAcavAmplitudes(self):
		'''
		Plots the max |Acav|^2 value for each LLE subsampled step. Useful when adding peturbations, quick and dirty way to see
		if the soliton starts to breathe or not or do some weird shit.
		'''
		maxAmps = np.max(np.abs(self.sol.Acav)**2,0)*1e3 #I'm pretty sure *1e3 is needed to keep units
		fig = go.Figure()
		tr = go.Scatter(y=maxAmps)
		#ALSO what is this square of Acav? I know its related to power but im confused. its getting late just get this to work
		fig.add_trace(tr)
		fig.update_layout(xaxis_title="LLE sub-sampled step", yaxis_title="Intra-cavity Power (mW)")
		return go.FigureWidget(fig)


	def plotIntraCavityPower(self):
		'''
		Plots intracavity power per detuning step.
		'''
		fig = go.Figure()
		tr = go.Scatter(y=self.sol.Pcomb * 1e3)
		fig.add_trace(tr)
		fig.update_layout(xaxis_title="LLE sub-sampled step", yaxis_title="Intra-cavity Power (mW)")
		return go.FigureWidget(fig)

	def plotAcav(self,rows=1,cols=1,startidx = 0,step=1):
		'''
		Plots a grid of rows*col |Acav|^2 plots. Starting and startidx and stepping by step.

		Params:
			rows - number of rows desired to plot
			cols - number of columns desired to plot
			startidx - starting index for plotting
			step - index step in between successive plots
		'''
		fig, ax = plt.subplots(rows,cols,figsize=(10,10))
		idx = startidx
		for x in range(rows):
			for y in range(cols):
				ax[x][y].plot(np.abs(self.sol.Acav[:,idx])**2)
				idx+=step


	def plotSolitonPos(self):
		'''
		Plots Soliton position per time step in the cavity. (Theta vs time step)
		'''
		fig = go.Figure()
		x = np.arange(0, 2000)
		y = np.linspace(0,2*np.pi, self.sol.Acav.shape[0])
		xsteps = 1
		ysteps = 1
		tr = go.Heatmap(x= x[::xsteps], y = y[::ysteps],  z = np.abs(self.sol.Acav[::ysteps, ::xsteps])**2)
		fig.add_trace(tr)
		fig.update_layout(xaxis_title = "LLE subsampled step", yaxis_title = "Resonator angle θ (x π)")
		return go.FigureWidget(fig)


	def plotDint(self):
		return self.Analyze(plot=True)

	def plotThetaPerStep(self):
		'''
		Plots the unwraped theta values per time step
		'''
		pass


'''
Probably could do without doing Analyze and Setup(?) for every simulation. Look into to it to make
sure when its needed and when it is redundant
'''
