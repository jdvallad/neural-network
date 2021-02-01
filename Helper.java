import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Helper {
    public static double[] convertColorImageToDoubleList(ImageIcon icon) {
        BufferedImage image = (BufferedImage) icon.getImage();
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        List<Double> result = new ArrayList<>();
        for (int pixel : pixels)
            result.add((double) (pixel & 0xff) / 255.0);
        return toArray(result);
    }
    public static double[] toArray(List<Double> list){
        double[] res = new double[list.size()];
        for(int i = 0; i < res.length;i++)
            res[i] = list.get(i);
        return res;
    }
    public static ImageIcon convertDoubleListToColorImage(double[] dubs, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] outPixels = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        final int pixelLength = 3;
        for (int pixel = 0; pixel + 2 < dubs.length; pixel += pixelLength) {
            int res = 0;
            res |= (((int) (dubs[pixel] * 255.0)) & 0xff); // blue
            res |= ((((int) (dubs[pixel + 1] * 255.0)) & 0xff) << 8); // green
            res |= ((((int) (dubs[pixel + 2] * 255.0)) & 0xff) << 16); // red
            outPixels[pixel / 3] = res;
        }
        return new ImageIcon(out);
    }

    public static List<Double> convertGrayScaleImageToDoubleList(ImageIcon icon) {
        BufferedImage image = (BufferedImage) icon.getImage();
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        List<Double> result = new ArrayList<>();
        for (int i = 0; i + 2 < pixels.length; i += 3) {
            int temp = 0;
            temp += pixels[i];
            temp += pixels[i + 1];
            temp += pixels[i + 2];
            result.add ( (double) temp / 765.0);
        }
        return result;
    }

    public static ImageIcon convertDoubleListToGrayScaleImage(double[] dubs, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] outPixels = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        for (int pixel = 0; pixel< dubs.length; pixel ++) {
            int res = 0;
            res |= (((int) (dubs[pixel] * 255.0)) & 0xff); // blue
            res |= ((((int) (dubs[pixel] * 255.0)) & 0xff) << 8); // green
            res |= ((((int) (dubs[pixel] * 255.0)) & 0xff) << 16); // red
            outPixels[pixel] = res;
        }
        return new ImageIcon(out);
    }

    public static ImageIcon imageFromFilePath(String filePath) throws IOException {
        return new ImageIcon(ImageIO.read(new File(filePath)));
    }

    public static double sigmoid(double x, boolean derivative) {
        if (!derivative)
            return 1. / (1. + Math.exp(-x));
        else {
            double k = sigmoid(x, false);
            return k * (1. - k);
        }
    }

    public static double relu(double x, boolean derivative) {
        if (!derivative)
            return Math.max(0., x);
        else
            return x > 0. ? 1. : 0.;
    }

    public static double dot(double[] x, double[] y) {
        double sum = 0.;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }
    public static double zed(double[] x, double[] y, double b){
        return b + dot(x,y);
    }
    public static double cost(double x, double y, boolean derivative){
        if (!derivative)
            return Math.pow(x - y,2);
        else
            return 2. * (x - y);
    }
    public static void clear(double[] arr, double base){
        Arrays.fill(arr, base);
    }
}
