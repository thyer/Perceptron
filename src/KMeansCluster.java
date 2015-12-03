public class KMeansCluster {
	private final boolean KMEDOIDS = false;
	private Matrix instances;
	private double[] centroid;

	public KMeansCluster(double[] centroid, Matrix m) {
		this.centroid = centroid;
		this.instances = m;
	}

	public void addInstance(Matrix m, int rowStart, int colStart, int rowCount) throws Exception {
		this.instances.add(m, rowStart, colStart, rowCount);
		
	}
	
	public boolean containsInstance(double[] instance){
		for(int i = 0; i < instances.rows(); ++i){
			if (instances.row(i) == instance){
				return true;
			}
		}
		return false;
	}
	
	public double calcAverageDissimilarity(double[] toCompare){
		double totalDissimilarity = 0;
		for (int r = 0; r < instances.rows(); ++r){
			double distance = 0;
			double[] instance = instances.row(r);
			for(int i = 0; i < instance.length; ++i){
				distance += Math.pow(toCompare[i] - instance[i], 2.0);
			}
			totalDissimilarity += Math.sqrt(distance);
		}
		return totalDissimilarity/instances.rows();
	}
	
	public Matrix getInstances() {
		return instances;
	}
	
	public void wipeInstances(){
		Matrix m = new Matrix(instances, 0, 0, 0, instances.cols());
		this.instances = m;
	}

	public void calcNewCentroid() {
		double[] newCentroid = null;
		
		//If no instances, stop
		if(this.getInstances().rows() < 1){
			return;
		}
		if(KMEDOIDS){
			double bestDistance = Double.MAX_VALUE;
			for(int i = 0; i < instances.rows(); ++i){
				double[] instance = instances.row(i);
				double distance = this.calcAverageDissimilarity(instance);
				if(distance < bestDistance){
					bestDistance = distance;
					newCentroid = instance;
				}
			}
			
		}
		else{
			//Initialize the new centroid to all zero
			newCentroid = new double[centroid.length];
			for(int i = 0; i < newCentroid.length; ++i){
				if(this.instances.valueCount(i) == 0){
					newCentroid[i] = this.instances.columnMean(i);
				}
				else{
					newCentroid[i] = this.instances.mostCommonValue(i);
				}
			}
		}
		
		//Finally, set the newCentroid to be the cluster's centroid
		this.centroid = newCentroid;
	}

	public double getDistance(double[] instance) {
		double totalDistance = 0.0;
		for(int i = 0; i < instance.length; ++i){
			if(instance[i] == Matrix.MISSING || centroid[i] == Matrix.MISSING){
				totalDistance += 1;
			}
			else if(instance[i] != this.centroid[i]){
				// Add squared difference if attribute is continuous, else add one
				if(this.instances.valueCount(i) == 0){
					totalDistance += Math.pow(instance[i] - centroid[i], 2.0);
				}
				else{
					totalDistance += 1;
				}
			}
		}
		return Math.sqrt(totalDistance);
	}

	public double[] getCentroid() {
		return this.centroid;
	}

	public double calcSSE() {
		double totalSSE = 0;
		for(int row = 0; row < instances.rows(); ++row){
			double[] instance = instances.row(row);
			totalSSE += Math.pow(this.getDistance(instance), 2.0);
		}
		return totalSSE;
	}

}
