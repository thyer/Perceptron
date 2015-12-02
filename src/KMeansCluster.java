import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class KMeansCluster {
	private Matrix instances;
	private double[] centroid;

	public KMeansCluster(double[] centroid, Matrix m) {
		this.centroid = centroid;
		this.instances = m;
	}

	public void addInstance(Matrix m, int rowStart, int colStart, int rowCount) throws Exception {
		this.instances.add(m, rowStart, colStart, rowCount);
		
	}
	
	public Matrix getInstances() {
		return instances;
	}
	
	public void wipeInstances(){
		Matrix m = new Matrix(instances, 0, 0, 0, instances.cols());
		this.instances = m;
	}

	public void calcNewCentroid() {
		//System.out.println("Calculating new centroid...");
		//If no instances, stop
		if(this.getInstances().rows() < 1){
			//System.out.println("No instances, returning");
			return;
		}
		//Initialize the new centroid to all zero
		double[] newCentroid = new double[centroid.length];
		for(int i = 0; i < newCentroid.length; ++i){
			if(this.instances.valueCount(i) == 0){
				newCentroid[i] = this.instances.columnMean(i);
			}
			else{
				newCentroid[i] = this.instances.mostCommonValue(i);
			}
		}
		
		//Finally, set the newCentroid to be the cluster's centroid
		this.centroid = newCentroid;
		String output = "[";
		for(int i = 0; i < centroid.length; ++i){
			if(centroid[i] == Matrix.MISSING){
				output+= "?";
			}
			else{
				output+= Math.round(centroid[i]*1000d)/1000d;
			}
			if(i < centroid.length -1){
				output += ", ";
			}
			else{
				output+= "]";
			}
		}
		//System.out.println(output);
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
