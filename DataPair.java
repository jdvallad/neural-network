import java.io.Serializable;
public class DataPair implements Serializable{
    public final double[] input;
    public final double[] expected;

    public DataPair(double[] input, double[] expected) {
        this.input = input;
        this.expected = expected;
    }
}
