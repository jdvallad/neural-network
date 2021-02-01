import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {
    public List<Layer> layers = new ArrayList<>();

    public double[] compute(double[] input) {
        double[] temp = input.clone();
        for (Layer l : layers)
            temp = l.feedForward(temp, false);
        return temp;
    }

    public void add(Layer layer) {
        layers.add(layer);
        int i = layers.size() - 1;
        if (i > 0) {
            layers.get(i).priorLayer = layers.get(i - 1);
            layers.get(i - 1).nextLayer = layers.get(i);
        }
    }

    public double cost(double[] output, double[] expected) {
        double sum = 0.;
        for (int i = 0; i < output.length; i++)
            sum += Helper.cost(output[i], expected[i], false);
        return sum / ((double) expected.length);
    }

    public IEPair train(DataIterator data) throws Exception {
        if (!data.hasNextBatch())
            throw new Exception();
        List<IEPair> pairs = data.nextBatch();
        double[] temp = null;
        for (IEPair pair : pairs) {
            temp = pair.input.clone();
            for (Layer l : layers)
                temp = l.feedForward(temp, true);
        }
        for (Layer l : layers) {
            l.nudge(l.weightGradients, l.biasGradients, -1 / ((double) data.batchSize));
            l.resetGradients();
        }
        return pairs.get(0);
    }

    public void save(String filePath) {
        try {
            FileOutputStream fileOut =
                    new FileOutputStream(filePath);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
        } catch (IOException i) {
            i.printStackTrace();
        }
    }

    public static NeuralNetwork load(String filePath) {
            NeuralNetwork e = null;
            try {
                FileInputStream fileIn = new FileInputStream(filePath);
                ObjectInputStream in = new ObjectInputStream(fileIn);
                e = (NeuralNetwork) in.readObject();
                in.close();
                fileIn.close();
            } catch (IOException | ClassNotFoundException i) {
                i.printStackTrace();
            }
            return e;
        }
}
