import os,sys,bpy
import numpy as np
sys.path.append(os.getcwd())
import graft as gf
import quaternion as qt
import utils as ut
#---------------------------------------------------------------------------------------#
# Helper functions
#-----------------------------------------------------------------------------#
def ravel(arr):
	return np.array( [outer for inner in arr for outer in inner] )
#---------------------------------------------------------------------------------------#
def dict_prop(prop, header, data):
	data = np.array(data)
	return data[:, list(header).index(prop)]
#---------------------------------------------------------------------------------------#
def altPath(path, newExt, locType="Front", addStr=None):
	locDict = { "Front" : 0, "End" : -1}
	dirList = path.split(os.sep)
	filename = dirList[-1].split('.')
	loc = locDict[locType]
	filename[loc] = newExt.strip('.')
	if addStr != None:
		filename[loc-1] += addStr
	dirList[-1] = '.'.join(filename)
	return os.sep.join(dirList)
#---------------------------------------------------------------------------------------#
# Calculate the total number of times "run" command has been invoked in the simulation
#---------------------------------------------------------------------------------------#
def runCount(filename, key):
	cval = 0; header = []
	with open(filename) as inFile:
		for line in inFile:
			_head = line.split()
			if key in _head:
				cval += 1
				header = np.union1d(_head, header)
	return cval, header
#---------------------------------------------------------------------------------------#
# Read standard LAMMPS output file (log.lammps/thermo output)
#---------------------------------------------------------------------------------------#
def read(infile, start, stop='Loop'):
	lines = []
	count = 0
	for line in infile:
		try:
			header = line.split()
			if start in header:
				break
		except IndexError:
			continue

	for line in infile:
		try:
			if line.split()[0] == stop:
				break
		except IndexError:
			break
		try:
			lines.append([float(x) for x in line.split()])
		except ValueError:
			count += 1
			continue
	if count > 0:
		return lines, header, count, 1
	else:
		return lines, header, count, 0
#---------------------------------------------------------------------------------------#
def get_box(datafile, time_series):
	filename = altPath(datafile, 'out')
	# Count the total number of runs in the file
	cval, header = runCount(filename, "TotEng")
	# Empty arrays
	dList = np.zeros([1,len(header)]); runData = []
	gWarn = 0; gCount = 0; RC = 0
	with open(filename) as infile:
		for c in np.arange(cval):
			RC += 1
			data, _head, count, warn = read(infile, "TotEng")
			gWarn = (warn or gWarn)
			gCount += count
			runData.append(len(data)-1)
			# Find the indices as corresponding to the global header list
			ind = np.array( [ np.where(header==x)[0][0] for x in _head ] )
			# Create an empty array
			dapp = np.zeros( [len(data),len(header) ] )
			dapp[:,ind] = np.stack(data)
			if RC == 0:
				dList = dapp
			else:
				dList = np.r_[dList, dapp[1:,:]]
	if gWarn == 1:
		print("WARNING: %d warnings found in file, " % gCount, filename)
	tList = dict_prop('Time', header, dList)
	mask = np.in1d(tList, time_series)
	modList = dList[mask]
	time = dict_prop("Time", header, modList)
	xlo = dict_prop("Xlo", header, modList)
	xhi = dict_prop("Xhi", header, modList)
	ylo = dict_prop("Ylo", header, modList)
	yhi = dict_prop("Yhi", header, modList)
	zlo = dict_prop("Zlo", header, modList)
	zhi = dict_prop("Zhi", header, modList)
	return time, np.c_[xlo, xhi, ylo, yhi, zlo, zhi], runData
#-----------------------------------------------------------------------------#
def oct_q():
	'''
	Calculate the quaternions for chiral octahedral symmentry
	Left-hand thumb rule x-cross-y = z: only 24 combinations out of the 48 possible
	'''
	ec1, ec2, ec3 = np.eye(3).T
	en1, en2, en3 = -1*np.eye(3).T
	qd = []
	# For x-local as x-global
	qd.append( qt.from_rotation_matrix(np.c_[ec1, ec2, ec3]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec1, en2, en3]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec1, ec3, en2]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec1, en3, ec2]) )
	qd.append( qt.from_rotation_matrix(np.c_[en1, ec2, en3]) )
	qd.append( qt.from_rotation_matrix(np.c_[en1, en2, ec3]) )
	qd.append( qt.from_rotation_matrix(np.c_[en1, ec3, ec2]) )
	qd.append( qt.from_rotation_matrix(np.c_[en1, en3, en2]) )
	# For y-local as x-global
	qd.append( qt.from_rotation_matrix(np.c_[ec2, ec3, ec1]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec2, en3, en1]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec2, ec1, en3]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec2, en1, ec3]) )
	qd.append( qt.from_rotation_matrix(np.c_[en2, ec3, en1]) )
	qd.append( qt.from_rotation_matrix(np.c_[en2, en3, ec1]) )
	qd.append( qt.from_rotation_matrix(np.c_[en2, ec1, ec3]) )
	qd.append( qt.from_rotation_matrix(np.c_[en2, en1, en3]) )
	# For z-local as x-global
	qd.append( qt.from_rotation_matrix(np.c_[ec3, ec1, ec2]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec3, en1, en2]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec3, ec2, en1]) )
	qd.append( qt.from_rotation_matrix(np.c_[ec3, en2, ec1]) )
	qd.append( qt.from_rotation_matrix(np.c_[en3, ec1, en2]) )
	qd.append( qt.from_rotation_matrix(np.c_[en3, en1, ec2]) )
	qd.append( qt.from_rotation_matrix(np.c_[en3, ec2, ec1]) )
	qd.append( qt.from_rotation_matrix(np.c_[en3, en2, en1]) )
	return np.array( qd )
#-----------------------------------------------------------------------------#
def min_alpha(qVec, target):
	'''
	vec:		vector to point along the vertical (y-) axis
	target:	(theta, psi) target orientation; e.g. {100}:(90.0, 90.0)
	'''
	eDict = {	"100" : [90., 90.] , "110" :  [90., 135.], "111": [54.7356, 135.]	}
	# Normalize and convert qVec to vec
	vec = qt.as_float_array( qVec/[ x.abs() for x in qVec] )[:,1:]
	# Convert to starting Euler angles
	theta = np.arccos(vec[:,1])
	psi = np.arctan2(vec[:,2], vec[:,0])
	# Corresponding starting quaternion values
	qet =	e2q( np.c_[np.zeros(len(vec)), theta, psi] )
	q_oct = oct_q()
	e24 = np.array([q2e(x*q_oct)[:,1:] for x in qet])
	alpha = []
	for trg in target:
		deul = np.cos( 0.5 * ( e24 - np.pi/180.0*np.array(eDict[trg]) ) )
		# Target orientation, c
		acal = 2.0 * np.arccos( deul[:,:,0] * deul[:,:,1] )
		acal[acal>np.pi] = 2*np.pi - acal[acal>np.pi]
		alpha.append( acal.min(axis=1) )
	return np.array( alpha ).T
#---------------------------------------------------------------------------------------#
def e2q(euler):
	'''
	Convert y-z'-y" Euler angles to quaternion
	'''
	cos = np.cos(euler); sin = np.sin(euler);
	# sin/cos of (phi+psi)
	yang = euler[:,0]+euler[:,2]
	# qr = 0.5 * np.sqrt(1 + tr(R))
	qr = 0.5 * np.sqrt( (1+cos[:,1]) * (1+np.cos(yang)) )
	qx = sin[:,1] * (sin[:,0] - sin[:,2])
	qy = (1+cos[:,1]) * np.sin(yang)
	qz = sin[:,1] * (cos[:,0] + cos[:,2])
	q = np.c_[qr, qx, qy, qz]
	q[:,1:] *= 0.25 / qr[:,np.newaxis]
	return qt.as_quat_array(q)
#-----------------------------------------------------------------------------#
def q2e(quat):
	'''
	Convert quaternion to y-z'-y" Euler angles
	Expects np/qt.quaternion input
	'''
	q = qt.as_float_array(quat)
	psi = np.arctan2( (q[:,3]*q[:,2] - q[:,1]*q[:,0]), (q[:,3]*q[:,0] + q[:,1]*q[:,2]) )
	theta = np.arccos( q[:,0]*q[:,0] + q[:,2]*q[:,2] - q[:,1]*q[:,1] - q[:,3]*q[:,3] )
	theta[ np.isnan(theta) ] = 0.0
	phi = np.arctan2( (q[:,3]*q[:,2] + q[:,1]*q[:,0]), (q[:,3]*q[:,0] - q[:,1]*q[:,2]) )

	phi[phi<0] += 2*np.pi
	psi[psi<0] += 2*np.pi
	return np.c_[phi, theta, psi]
#-----------------------------------------------------------------------------#
# Generate block maps
#-----------------------------------------------------------------------------#
def maps(nbx, nbz):
	# Generate list of all indices [0,#NP-1] in the block and its neighbors
	# 4 Neighboring blocks are (0,+1),(+1,+1),(+1,0),(+1,-1)
	neigh = lambda ix, iz: np.array( [ [ix, iz+1], [ix+1, iz+1], [ix+1, iz], [ix+1, iz-1] ] , dtype=int)
	# Number of blocks array
	NBL = np.array([nbx,nbz])
	irList = []; imFlag = []

	for ir in range(nbx*nbz):
		ix = ir%nbx
		iz = ir//nbx
		# Generate 4 neighbors as ix,iz pair
		nTemp = neigh(ix,iz)
		# Save image flags
		imFlag.append( nTemp // NBL )
		# Wrap by periodic boundaries
		nList = (nTemp + NBL) % NBL
		# Convert back to column-major format
		irList.append( nList[:,0] + nList[:,1]*nbx )

	return np.array(irList), np.array(imFlag)
#---------------------------------------------------------------------------------------#
# Molecular vector output file:
# Each block of data (representing a timestep) is demarcated by a header line
#---------------------------------------------------------------------------------------#
class MolVec:
	def __init__(self, filename, start=None, skip=None, end=None):
		self.filename = filename
		self._genLoc()
		self._setmask(start, skip, end)
#---------------------------------------------------------------------------------------#
	# mask = nskip = (# of timesteps to skip + 1)
	def _setmask(self, start=None, skip=None, end=None):
		conv = lambda x, y: y if (x == None) else x
		self._start = conv(start,0)
		self._skip = conv(skip,1)
		self._end = conv(end,len(self._data))
		assert (type(self._start) == int) and (type(self._skip) == int) and (type(self._end) == int)
		self.headMask = np.arange(self._start,self._end,self._skip)
		# re-/initialize all variables here
#---------------------------------------------------------------------------------------#
	def time(self):
		return self._time[self.headMask]
#---------------------------------------------------------------------------------------#
	# Generate the header and rawData of the file
	def _genLoc(self):
		with open(self.filename) as inFile:
			# Ignore first 3 title lines
			self.rawData = np.array( [ np.array( x.split() ).astype(float) for x in inFile.readlines() [3:] ] )
		# Collect location of header rows from the array
		# "header" rows: Timestep, nrows
		headLoc = []; lNo = 0
		while True:
			if lNo < len(self.rawData):
				headLoc.append( lNo )
				lNo += int(self.rawData[lNo][1]) + 1
			else:
				break
		self.headLoc = np.array(headLoc)
		# Confirm that number of chunks (molecules) does not change during the simulation
		nChunk = list( set(self.headLoc[1:]-self.headLoc[:-1]) )
		assert len( nChunk ) == 1
		self.nChunk = nChunk[0] - 1
		# Generate time list
		self._time = np.stack(self.rawData[self.headLoc])[:,0]
		# Indices of the rawData after removing the headLoc
		ind = np.setdiff1d(np.arange(len(self.rawData)), self.headLoc)
		arr = np.stack(self.rawData[ind])
		self._data = arr.reshape(-1,self.nChunk,arr.shape[1])
#---------------------------------------------------------------------------------------#
	# Generate a single snapshot (time) as a numpy array slice
	def _genStep(self, step):
		return self._data[step]
#---------------------------------------------------------------------------------------#
	# Generate data for a single molecule
	def genMol(self, molID, col):
		assert molID <= self.nChunk
		return self._data[self.headMask,molID, col]
#---------------------------------------------------------------------------------------#
	# Generate a single snapshot (time) as a numpy array slice
	def data(self):
		self.com = self._data[self.headMask,:,11:14]
		self.eul = self._data[self.headMask,:,1:4]*np.pi/180.0
		return self._data[self.headMask]
#---------------------------------------------------------------------------------------#
	# Associate a bond vector
	def bonds(self):
		self.bonds = BondVec(self)
		return self.bonds
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
class BondVec:
	def __init__(self, comVec, shape="cuboctahedron", beads=5, diameter=6.2, time_scale=10):
		if "graft" not in sys.modules:
			import graft as gf
		# Initialize the box vectors
		outFile = altPath(comVec.filename, "out.", locType="Front")
		time_series = time_scale * comVec.time()
		scaledTime, bbox, runData = get_box(outFile, time_series)
		dims = bbox[:,[1,3,5]] - bbox[:,[0,2,4]]
		# Calculate indices where box dimensions (parallel to the interface) don't change
		BVEC = np.sqrt(np.sum(dims[:,[0,2]]**2, axis=1))
		self.limit = (BVEC-BVEC[-1] < 1e-10)
		self.bbox = bbox[self.limit]; self.dims = dims[self.limit];
		'''
		Indices:	molID(0), Global-Euler(1-3), Local-Euler(4-6), alpha(7), AOR(8-10)
							COM(11-13), bodyID(14), efrac(15), active(16)
		'''
		com = comVec.data()[:,:,11:14]
		eul = comVec.data()[:,:,1:4]*np.pi/180.0
		self.com = com[self.limit]; self.eul = eul[self.limit]
		# Set the NP type
		self.set(shape=shape, beads=beads, diameter=diameter)
#-----------------------------------------------------------------------------#
# Reset the NP shape
#-----------------------------------------------------------------------------#
	def set(self, shape, beads, diameter):
		# Make arbitrary polygons
		_shape = getattr(gf, shape)
		poly = _shape(beads=beads, diameter=diameter)
		rmin, rmax = poly._extent()
		# min/max distance (between COMs)
		lbx = lbz = 1.5*rmax*2
		self.UL = 1.15*rmax*2
		self.LL = 0.95*rmin*2
		# Block-IDs along X and Z
		nbx = ( (self.dims[:,0]) / lbx ).astype(int)
		nbz = ( (self.dims[:,2]) / lbz ).astype(int)
		assert ( len(set(nbx))==1 and (len(set(nbz))==1) )
		self.nbx = list(set(nbx))[0]; self.nbz = list(set(nbz))[0];
		self.irMap, self.imFlag = maps( self.nbx, self.nbz )
		#print "IRLIST = ", irMap, "IMFLAG = ", imFlag
		ix = self.com[:,:,0] - self.bbox[:,np.newaxis,0]
		# Similarly,
		iz = self.com[:,:,2] - self.bbox[:,np.newaxis,4]
		# Wrap into periodic boundaries
		ix = ( (ix + self.dims[:,np.newaxis,0]) % self.dims[:,np.newaxis,0] // lbx).astype(np.int)
		iz = ( (iz + self.dims[:,np.newaxis,2]) % self.dims[:,np.newaxis,2] // lbz).astype(np.int)
		ix[ix >= self.nbx] = self.nbx - 1
		iz[iz >= self.nbz] = self.nbz - 1
		self.ir = ix + iz*self.nbx
		print (self.ir.shape), "ir = ", self.com[np.where(self.ir==-1)]
#---------------------------------------------------------------------------------------#
	# Lower limit where the box is not changing
#---------------------------------------------------------------------------------------#
	def low(self):
		return np.nonzero(self.limit)[0][0]
#---------------------------------------------------------------------------------------#
	# Save number of particles at distances between LL and UL from NP center
#---------------------------------------------------------------------------------------#
	def pair(self, frame, target=["100", "110"]):
		# Frame refers to corresponding timestep in MolVec; Convert to reduced steps
		if not self.limit[frame]:
			raise ValueError("Box size still changing. Pick frame >%d" % self.low() )
		ti = np.where(np.nonzero(self.limit)[0] == frame)[0][0]
		comT = self.com[ti]
		# Generate I-J list
		owned = [ list(np.where(self.ir[ti]==x)[0]) for x in range(self.nbx*self.nbz) ]
		# All NPs distributed into groups
		assert sum( [ len(x) for x in owned ] ) == len(self.ir[ti])
#-----------------------------------------------------------------------------#
		indices = []
		for irm in self.irMap:
			nind = []
			for neigh in irm:
				nind += owned[neigh]
			indices.append(nind)
#-----------------------------------------------------------------------------#
		# Inter-block pairs: Block with neighbors
		pair_i = ravel([ np.repeat(_own,len(indices[ib])) for ib, _own in enumerate(owned) ])
		pair_j = ravel([ np.tile(indices[ib],len(_own)) for ib, _own in enumerate(owned) ])
		# Intra-Block pairs
		_triu = lambda arr,i: np.concatenate([ np.array(x)[np.triu_indices(len(x),1)[i]] for x in arr if len(x) not in [0,1] ]).ravel()
		pair_i = np.r_[pair_i, _triu(owned,0)]
		pair_j = np.r_[pair_j, _triu(owned,1)]
		# Nearest-image convention
		RIJ = comT[pair_j,:] - comT[pair_i,:]
		RIJ = RIJ - np.rint(RIJ/self.dims[ti])*self.dims[ti]
#-----------------------------------------------------------------------------#
		# Quaternion-based bond-order parameter
#-----------------------------------------------------------------------------#
		# Quaternion for global rotation of the NP
		qg = np.array(  e2q(self.eul[ti]) )
		qg /= np.array( [x.abs() for x in qg] )
		# Corresponding Inverse quaternion
		qginv = np.array( [ x.inverse() for x in qg ] )
		# "Rotated" bond vector in quaternion form
		qRIJ = qt.as_quat_array(np.c_[ np.zeros(len(RIJ)), RIJ ] )
		# Calculate "un"-rotated bond vectors
		vec_i = qginv[pair_i]*qRIJ*qg[pair_i]
		vec_j = -1*qginv[pair_j]*qRIJ*qg[pair_j]
		#
		acorn = np.arccos(1.0/np.sqrt(3.0))
		alpha_norm = np.array( [ acorn, 0.25*np.pi ] )
		alpha_i = min_alpha(vec_i, target) 
		alpha_j = min_alpha(vec_j, target)
		axb = np.einsum("ij,ij->i", alpha_i, alpha_j)# / (alpha_norm**2)
		# Threshold of 5 degrees
		thresh = 10*np.pi/180.0
		adot = (alpha_i < thresh) & (alpha_j < thresh)
#-----------------------------------------------------------------------------#
		return pair_i, pair_j, RIJ, axb, adot
#---------------------------------------------------------------------------------------#
# Blender specific functions
#------------------------------------------------#
# Add single polyhedron
#------------------------------------------------#
class MESH_OT_primitive_polyhedron(bpy.types.Operator):
	"""Create a Polyhedron object"""
	bl_idname = "mesh.primitive_polyhedron"
	bl_label = "Polyhedron"
	bl_options = {"REGISTER", "UNDO"}
#------------------------------------------------#    
	poly_type: bpy.props.StringProperty(
		name="Type",
		description="Polyhedron type",
		default="Cuboctahedron",
	)
	beads: bpy.props.IntProperty(
		name="Beads",
		description="Number of beads per edge",
		default=5,
		min=2, soft_max=10,
	)
	diameter: bpy.props.FloatProperty(
		name="Bead diameter",
		description="Diameter of each bead",
		default=6.2,
		min=0.1, soft_max=10,
	)
	col_name: bpy.props.StringProperty(
		name="Collection",
		description="Collection name",
		default=bpy.context.collection.name,
	)
#------------------------------------------------#
	def execute(self, context):
		# Shade smooth
		bpy.ops.object.shade_smooth()
		# Create sample polyhedron instance
		print(self.poly_type)
		_shape = getattr(gf,"_".join(self.poly_type.lower().split()))
		poly = _shape(beads=self.beads, diameter=self.diameter)
		# Add the polyhedra to the collection
		tPol = self.add_mesh(poly)
		# Facet-specific material types
		matDict = { '100':0, '111':1, '110':2 }
		red = ut.basicMaterial((255,0,0), mat_name="Red")
		tPol.data.materials.append(red)
		green = ut.basicMaterial((0,255,0), mat_name="Green")
		tPol.data.materials.append(green)
		yellow = ut.basicMaterial((255,255,0), mat_name="Yellow")
		tPol.data.materials.append(yellow)
		# Assign material to facets
		for fi, face in enumerate(tPol.data.polygons):
			face.material_index = matDict[poly.miller[fi]]
		return {"FINISHED"}
#------------------------------------------------#
	@classmethod
	def poll(cls, context):
		#return (context.area.type == "VIEW_3D")
		return True
#------------------------------------------------#
	def add_mesh(self, poly):
		mesh = bpy.data.meshes.new("%s_mesh" % poly.sub)
		mesh.from_pydata(poly.vertices,[],poly.facets)
		obj = bpy.data.objects.new(poly.sub, mesh)
		xcol = bpy.data.collections.get(self.col_name)
		xcol.objects.link(obj)
		bpy.context.view_layer.objects.active = obj
		return obj
#------------------------------------------------#
'''
Adapter file for *.lammps averaged file
per-(rigid) body info imported from parser module
'''
class MESH_OT_poly_frame(bpy.types.Operator):
	"""Frame of trajectory of Polyhedra"""
	bl_idname = "mesh.poly_frame"
	bl_label = "Frame"
	bl_options = {"REGISTER", "UNDO"}
#------------------------------------------------#    
	poly_type: bpy.props.StringProperty(
		name="Type",
		description="Polyhedron type",
		default="Cuboctahedron",
		)
	trajectory: bpy.props.StringProperty(
		name="Input",
		description="Input trajectory file path",
		)
	frame: bpy.props.IntProperty(
		name="Frame",
		description="Frame number",
		default=-1,
		)
	col_name: bpy.props.StringProperty(
		name="Collection",
		description="Collection name",
		default="Polyhedra",
	)
#------------------------------------------------#
	def execute(self, context):
		# Create a collection if it doesn't exist
		col = ut.mkcol(self.col_name)
		# Extract location and orientation data per-body
		mol = MolVec(self.trajectory)
		com = mol.data()[self.frame,:,11:14]
		eul = mol.data()[self.frame,:,1:4]*np.pi/180.0
		quat = qt.as_float_array(e2q(eul))
		# Loop over all bodies
		print(self.poly_type)
		for ib in np.arange(mol.nChunk):
			bpy.ops.mesh.primitive_polyhedron(poly_type=self.poly_type, col_name=self.col_name)
			print(com[ib].shape)
			bpy.context.view_layer.objects.active.location = com[ib]
			bpy.context.view_layer.objects.active.rotation_mode = 'QUATERNION'
			bpy.context.view_layer.objects.active.rotation_quaternion = quat[ib]
		return {"FINISHED"}
#------------------------------------------------#
	@classmethod
	def poll(cls, context):
	#return (context.area.type == "VIEW_3D")
		return True
#------------------------------------------------#
'''
Wrapper function for MESH_OT_poly_frame
Adds all selected frames from trajectory file
'''
class MESH_OT_poly_trajectory(bpy.types.Operator):
	"""Trajectory of Polyhedra simulation"""
	bl_idname = "mesh.poly_trajectory"
	bl_label = "Trajectory"
	bl_options = {"REGISTER", "UNDO"}
#------------------------------------------------#    
	poly_type: bpy.props.StringProperty(
		name="Type",
		description="Polyhedron type",
		default="Cuboctahedron",
		)
	trajectory: bpy.props.StringProperty(
		name="Input",
		description="Input trajectory file path",
		)
	traj_start: bpy.props.IntProperty(
		name="Start",
		description="Starting frame of the trajectory",
		default=1200,
		)
	traj_skip: bpy.props.IntProperty(
		name="Skip",
		description="Skip frames of the trajectory",
		default=10,
		)
	col_name: bpy.props.StringProperty(
		name="Collection",
		description="Collection name",
		default="Polyhedra",
	)
	drawbox: bpy.props.BoolProperty(
		name="DrawBox",
		description="Draw bounding box",
		default=True,
		)
#------------------------------------------------#
	def execute(self, context):
		# Create a collection if it doesn't exist
		col = ut.mkcol(self.col_name)
		# Extract location and orientation data per-body
		print(self.poly_type)
		mol = MolVec(self.trajectory, start=self.traj_start, skip=self.traj_skip)
		com = mol.data()[:,:,11:14]
		eul = mol.data()[:,:,1:4]*np.pi/180.0
		# Generate template polyhedra
		bpy.ops.mesh.primitive_polyhedron(poly_type=self.poly_type, col_name=self.col_name)
		tPol = bpy.context.view_layer.objects.active
		body = [tPol]
		# Generate rest of the bodies and store in a list
		for ib in np.arange(1,mol.nChunk):
			copy = tPol.copy()
			copy.data = copy.data.copy()
			obj = copy
			body.append(obj)
		# Output frames per second (FPS)
		out_fps = 24
		# Trajectory FPS i.e. number of trajectory frames per second
		traj_fps = 24
		# Add keyframe every fskip frames
		f_skip = int(out_fps/traj_fps)
		# Set first and last frame index
		f_start = 0; f_end = (len(mol.time())-1)*f_skip;
		bpy.context.scene.frame_start = f_start
		bpy.context.scene.frame_end = f_end
		# Loop over all frames
		for fi, frame in enumerate(mol.time()):
			bpy.context.scene.frame_set(fi*f_skip)
			quat = qt.as_float_array(e2q(eul[fi]))
			# Loop over all bodies
			for ib in np.arange(mol.nChunk):
				obj = body[ib]
				obj.location = com[fi][ib]
				# Insert new keyframe for "location"
				obj.keyframe_insert(data_path="location")
				obj.rotation_mode = 'QUATERNION'
				obj.rotation_quaternion = quat[ib]
				# Similarly, for "rotation"
				obj.keyframe_insert(data_path="rotation_quaternion")

		# Add to collection/scene
		for ob in body[1:]:
			col.objects.link(ob)
		#if self.drawbox:
		#	bbox = drawbox(bond.dims, thickness=5.0, color=(0,0,0) )
		return {"FINISHED"}
#------------------------------------------------#
	@classmethod
	def poll(cls, context):
	#return (context.area.type == "VIEW_3D")
		return True
#------------------------------------------------#
'''
Adapter file for *.lammps averaged file
per-(rigid) body info imported from parser module
Draw only bonds per frame
'''
class MESH_OT_bond_frame(bpy.types.Operator):
	"""Frame of trajectory of bonds"""
	bl_idname = "mesh.bond_frame"
	bl_label = "Frame"
	bl_options = {"REGISTER", "UNDO"}
#------------------------------------------------#    
	trajectory: bpy.props.StringProperty(
		name="Input",
		description="Input trajectory file path",
		)
	frame: bpy.props.IntProperty(
		name="Frame",
		description="Frame number",
		default=-1,
		)
	radius: bpy.props.FloatProperty(
		name="Radius",
		description="Bond Radius",
		default=2.0,
		)
	col_name: bpy.props.StringProperty(
		name="Collection",
		description="Collection name",
		default="Bonds",
	)
	drawbox: bpy.props.BoolProperty(
		name="DrawBox",
		description="Draw bounding box",
		default=True,
		)
#------------------------------------------------#
	def execute(self, context):
		col_100 = ut.mkcol(self.col_name + "_100")
		col_110 = ut.mkcol(self.col_name + "_110")
		# Materials
		palette = [ (129,212,106), (208,252,126), (248,204,70), (235,51,61) ]
		mat100 = ut.basicMaterial(palette[3], mat_name="mat100")
		mat110 = ut.basicMaterial(palette[2], mat_name="mat110")

		# Initialize empty set
		bondSet = set()
		# Extract location and orientation data per-body
		mol = MolVec(self.trajectory, skip=100)
		bond = mol.bonds()
		pair_i, pair_j, RIJ, AXB, ADOT = bond.pair(self.frame)
		RMAG = np.sqrt(np.sum(RIJ**2, axis=1))
		xind = (RMAG < bond.UL) & ADOT[:,0]
		yind = (RMAG < bond.UL) & ADOT[:,1]
		cind = xind | yind
		print("XIND=", pair_i[xind], pair_j[xind])
		RIJ = RIJ[cind]; RMAG = RMAG[cind]; AXB = AXB[cind]
		pair_i = pair_i[cind]; pair_j = pair_j[cind];

		unit = RIJ/RMAG[:,np.newaxis]
		mid = mol.com[self.frame][pair_i] + 0.5*RIJ
		align = np.c_[ np.zeros(len(pair_i)), np.arccos(unit[:,2]), np.arctan2(unit[:,1],unit[:,0]) ]

		# Overall storage
		bex = np.empty( (mol.nChunk,mol.nChunk), dtype=np.object)
		# Create one template cylinder
		bpy.ops.mesh.primitive_cylinder_add(radius=self.radius, depth=1.0)
		tBond = bpy.context.view_layer.objects.active
		# Remove from global collection
		#bpy.data.collections["Collection"].objects.unlink(bpy.context.view_layer.objects.active)
		# For efficiency, replace high-level bpy.ops.mesh.* commands with low level code
		#obs = [tBond]

		for ib in np.arange(len(RIJ)):
			#bpy.ops.mesh.primitive_cylinder_add(radius=self.radius, location=mid[ib], depth=RMAG[ib])
			copy = tBond.copy()
			copy.data = copy.data.copy()
			#bpy.data.collections["Collection"].objects.link(copy)
			#obs.append(copy)
			#bex[pair_i[ib]-1, pair_j[ib]-1] = bpy.context.view_layer.objects.active
			bex[pair_i[ib]-1, pair_j[ib]-1] = copy

		# Loop over all bodies
		for ib in np.arange(len(RIJ)):
			obj = bex[pair_i[ib]-1, pair_j[ib]-1]
			print(bex.shape, obj)
			obj.location = mid[ib]
			obj.rotation_mode = 'XYZ'
			obj.rotation_euler = align[ib]
			obj.scale[2] = RMAG[ib]
			try:
				if ADOT[cind][ib,0]:
					col_100.objects.link(obj)
					obj.active_material = mat100
				if ADOT[cind][ib,1]:
					col_110.objects.link(obj)
					obj.active_material = mat110
			except RuntimeError:
				pass
		return {"FINISHED"}
#------------------------------------------------#
	@classmethod
	def poll(cls, context):
	#return (context.area.type == "VIEW_3D")
		return True
#------------------------------------------------#
'''
Wrapper function for MESH_OT_bond_frame
Adds all selected frames from trajectory file
'''
class MESH_OT_bond_trajectory(bpy.types.Operator):
	"""Trajectory of Polyhedra simulation: Bonds"""
	bl_idname = "mesh.bond_trajectory"
	bl_label = "Bond Trajectory"
	bl_options = {"REGISTER", "UNDO"}
#------------------------------------------------#    
	trajectory: bpy.props.StringProperty(
		name="Input",
		description="Input trajectory file path",
		)
	traj_start: bpy.props.IntProperty(
		name="Start",
		description="Starting frame of the trajectory",
		default=1200,
		)
	traj_skip: bpy.props.IntProperty(
		name="Skip",
		description="Skip frames of the trajectory",
		default=10,
		)
	radius: bpy.props.FloatProperty(
		name="Radius",
		description="Bond Radius",
		default=2.0,
		)
	col_name: bpy.props.StringProperty(
		name="Collection",
		description="Collection name",
		default="Bonds",
	)
	drawbox: bpy.props.BoolProperty(
		name="DrawBox",
		description="Draw bounding box",
		default=True,
		)
#------------------------------------------------#
	def execute(self, context):
		# New collection lists
		Arr100 = []; Arr110 = [];
		# And materials
		palette = [ (129,212,106), (208,252,126), (248,204,70), (235,51,61) ]
		mat100 = ut.basicMaterial(palette[3], mat_name="mat_100")
		mat110 = ut.basicMaterial(palette[2], mat_name="mat_110")

		# Extract location and orientation data per-body
		mol = MolVec(self.trajectory, start=self.traj_start, skip=self.traj_skip)
		#mol._setmask(skip=100)
		print("TIME", mol.headMask)
		bond = mol.bonds()
		# Overall storage
		b100 = np.empty( (mol.nChunk,mol.nChunk), dtype=np.object)
		b110 = np.empty( (mol.nChunk,mol.nChunk), dtype=np.object)
		# Create one template cylinder/Bond
		bpy.ops.mesh.primitive_cylinder_add(radius=self.radius, depth=1.0)
		# For efficiency, replace high-level bpy.ops.mesh.* commands with low level code
		tBond = bpy.context.view_layer.objects.active
		bpy.data.collections["Collection"].objects.unlink(tBond)
		tBond.rotation_mode = 'XYZ'

		# Output frames per second (FPS)
		out_fps = 24
		# Trajectory FPS i.e. number of trajectory frames per second
		traj_fps = 24
		# Add keyframe every fskip frames
		f_skip = int(out_fps/traj_fps)
		# Set first and last frame index
		f_start = 0; f_end = len(mol.time())*f_skip;
		bpy.context.scene.frame_start = f_start
		bpy.context.scene.frame_end = f_end
		# Loop over all frames
		for fi, tval in enumerate(mol.time()):
			print("XXX=", fi, bond.low())
			if fi <= bond.low():
				continue
			bframe = fi*f_skip
			bpy.context.scene.frame_set(bframe)
			#------------------------------------------------#
			pair_i, pair_j, RIJ, AXB, ADOT = bond.pair( fi )
			RMAG = np.sqrt(np.sum(RIJ**2, axis=1))
			xind = (RMAG < bond.UL) & ADOT[:,0]
			yind = (RMAG < bond.UL) & ADOT[:,1]
			cind = xind | yind
			print("XIND=", pair_i[xind], pair_j[xind])
			RIJ = RIJ[cind]; RMAG = RMAG[cind]; AXB = AXB[cind]
			pair_i = pair_i[cind]; pair_j = pair_j[cind];
			unit = RIJ/RMAG[:,np.newaxis]
			mid = mol.com[fi][pair_i] + 0.5*RIJ
			align = np.c_[ np.zeros(len(pair_i)), np.arccos(unit[:,2]), np.arctan2(unit[:,1],unit[:,0]) ]
			#------------------------------------------------#
			# Loop over all bodies
			for ib in np.arange(len(RIJ)):
				# Bond Type-100
				if ADOT[cind][ib,0]:
					PIJ = b100[pair_i[ib]-1, pair_j[ib]-1]
					xmat = mat100; xArr = Arr100;
				elif ADOT[cind][ib,1]:
					PIJ = b110[pair_i[ib]-1, pair_j[ib]-1]
					xmat = mat110; xArr = Arr110;
				# If bond doesn't exist yet, create new
				if PIJ is None:
					copy = tBond.copy()
					copy.data = copy.data.copy()
					PIJ = copy
					# Set bond-type properties
					xArr.append(PIJ)
					PIJ.active_material = xmat
					# Initialize Visibilty settings
					PIJ.animation_data_create()
					PIJ.animation_data.action = bpy.data.actions.new(name="Visibility")
					fcr = PIJ.animation_data.action.fcurves.new(data_path="hide_render")
					fcr.keyframe_points.add(1)
					# Repeat for viewport
					fcv = PIJ.animation_data.action.fcurves.new(data_path="hide_viewport")
					fcv.keyframe_points.add(1)
					# Visibilty is false upto this point
					fcr.keyframe_points[0].co = (bframe-1), True
					fcv.keyframe_points[0].co = (bframe-1), True
				#------------------------------------------------#
				# Set bond-specific properties
				#------------------------------------------------#
				# Visibility settings
				obj = PIJ
				fcr = obj.animation_data.action.fcurves.find("hide_render")
				fcv = obj.animation_data.action.fcurves.find("hide_viewport")
				if not fcv.keyframe_points[-1].co[0] == bframe:
					fcr.keyframe_points.add(2)
					fcv.keyframe_points.add(2)
					# Visibilty on for this frame only
					fcr.keyframe_points[-2].co = bframe, False
					fcv.keyframe_points[-2].co = bframe, False
				# Visibilty off from the next frame onwards
				fcr.keyframe_points[-1].co = (bframe+1), True
				fcv.keyframe_points[-1].co = (bframe+1), True
				#------------------------------------------------#
				obj.location = mid[ib]
				obj.rotation_euler = align[ib]
				# Bond length
				obj.scale[2] = RMAG[ib]
				# Insert new keyframe for "location"
				obj.keyframe_insert(data_path="location")
				# Similarly, for "rotation"
				obj.keyframe_insert(data_path="rotation_euler")

		# New collections
		coll100 = ut.mkcol(self.col_name + "_100")
		for ob in Arr100:
			try:
				coll100.objects.link(ob)
			except RuntimeError:
				pass
		coll110 = ut.mkcol(self.col_name + "_110")
		for ob in Arr110:
			try:
				coll110.objects.link(ob)
			except RuntimeError:
				pass

		if self.drawbox:
			bbox = drawbox(bond.dims, thickness=5.0, color=(0,0,0) )

		return {"FINISHED"}
#------------------------------------------------#
	@classmethod
	def poll(cls, context):
	#return (context.area.type == "VIEW_3D")
		return True
#------------------------------------------------#
# Blender-specific wrapping functions
#------------------------------------------------#
# Draw orthogonal bounding box updated every frame
def drawbox(dims, thickness=5.0, color=(0,0,0) ):
	xdim = dims[0][0]
	dims /= float(xdim)
	bpy.ops.mesh.primitive_cube_add( size=xdim, align='WORLD', location=(0,0,0) )
	obj = bpy.context.active_object
	obj.name = "BBox"
	wmod = obj.modifiers.new("Wire", type="WIREFRAME")
	wmod.thickness = thickness
	#	Add material
	matbox = ut.basicMaterial( color, mat_name="mat_box" )
	obj.active_material = matbox
	# Initialize scale settings
	obj.animation_data_create()
	obj.animation_data.action = bpy.data.actions.new(name="Scale")
	s_x = obj.animation_data.action.fcurves.new(data_path="scale", index=0)
	s_x.keyframe_points.add( len(dims) )
	s_z = obj.animation_data.action.fcurves.new(data_path="scale", index=2)
	s_z.keyframe_points.add( len(dims) )
	for di, dval in enumerate(dims):
		s_x.keyframe_points[di].co = di, dval[0]
		s_z.keyframe_points[di].co = di, dval[2]
	return obj
#------------------------------------------------#
_load_class = [ MESH_OT_primitive_polyhedron,
								MESH_OT_poly_frame,
								MESH_OT_poly_trajectory,
								MESH_OT_bond_frame,
								MESH_OT_bond_trajectory,
							]
#------------------------------------------------#
def register():
	for bclass in _load_class:
		bpy.utils.register_class(bclass)
#------------------------------------------------#
def unregister():
	for bclass in _load_class:
		bpy.utils.unregister_class(bclass)
#------------------------------------------------#
if __name__ == "__main__":
	register()
	# Shade smooth
	bpy.ops.object.shade_smooth()
