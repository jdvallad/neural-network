import java.io.Serializable;
import java.util.List;

public class IEPair implements Serializable{
    public final double[] input;
    public final double[] expected;
    public final boolean hasExpected;

    public IEPair(double[] input, double[] expected) {
        this.input = input;
        this.expected = expected;
        this.hasExpected = true;
    }
    public IEPair(double[] input) {
        this.input = input;
        this.expected = null;
        this.hasExpected = false;
    }
}
