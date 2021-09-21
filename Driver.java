import java.util.Map;

public class Driver {
    public static void main(String[] args) throws Exception {
        DataIterator genIterator = new mnistReverseIterator(100, 420);
        nn gen = new nn(10, "logLoss", "./serials/mnistDigitGenerate.ser");
        gen.add("sigmoid", 16);
        gen.add("leakyRelu", 16);
        gen.add("sigmoid", 28 * 28);
        gen.build();
        DataIterator disIterator = new mnistDisIterator(100, 420, gen); 
        nn dis = new nn(28 * 28, "logLoss", "./serials/mnistDigitDiscriminate.ser");
        dis.add("sigmoid", 16);
        dis.add("leakyRelu", 16);
        dis.add("sigmoid", 2);
        dis.build();
        Map<String, Object> input = Map.of("width", 28, "height", 28, "color", false, "scale", 10.);
        Map<String, Object> output = Map.of("width", 2, "height", 1, "color", false, "scale", 80.);
        dis.train(disIterator, 200, .1, true, input, output);
    }
}