import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class KMeansCluster {
	private List<double[]> instances;
	private double[] centroid;

	public KMeansCluster(double[] centroid) {
		this.centroid = centroid;
		this.instances = new ArrayList<double[]>();
	}

	public void addInstance(double[] row) {
		this.instances.add(row);
		
	}

	public List<double[]> getInstances() {
		return Collections.unmodifiableList(this.instances);
	}

	public void removeInstance(double[] instance) {
		this.instances.remove(instance);
		
	}

	public void calcNewCentroid() {
		// TODO Auto-generated method stub
		
	}

	public double getDistance(double[] instance) {
		// TODO Auto-generated method stub
		return 0;
	}

}
