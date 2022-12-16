import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;

public class ImageViewer {
    private final JFrame frame;
    private JLabel label;

    public ImageViewer(String s) {
        frame = new JFrame(s);
        label = new JLabel();
        frame.add(label);
        frame.setResizable(false);
    }

    public void show(String filePath, int width, int height) {
        frame.remove(label);
        ImageIcon image = new ImageIcon(filePath);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT))
        );
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void show(ImageIcon image, int width, int height) {
        frame.remove(label);
        label = new JLabel(
                new ImageIcon(image.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT))
        );
        frame.add(label);
        frame.pack();
        frame.validate();
        frame.repaint();
    }

    public void show(String filePath) {
        show(new ImageIcon(filePath));
    }

    public void show(ImageIcon image) {
        show(image, image.getIconWidth(), image.getIconHeight());
    }
  
    public void show(String filePath, double scale) {
        show(new ImageIcon(filePath), scale);
    }

    public void show(ImageIcon image, double scale) {
        show(image, (int) (image.getIconWidth() * scale), (int) (image.getIconHeight() * scale));
    }

    public void setVisible(boolean visible) {
        frame.setVisible(visible);
        frame.repaint();
    }

    private static double[] colorToList(ImageIcon icon) {
        BufferedImage image = (BufferedImage) icon.getImage();
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        double[] res = new double[3*pixels.length];
        for (int i = 0; i < res.length; i+=3){
            res[i] = (double) (pixels[i/3] & 0xff) / 255.0;
            res[i+1] = (double) ((pixels[i/3] >> 8) & 0xff) / 255.0;
            res[i+2] = (double) ((pixels[i/3] >> 16) & 0xff) / 255.0;
            }
        return res;
    }

    private static ImageIcon listToColor(double[] dubs, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] outPixels = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        final int pixelLength = 3;
        for (int pixel = 0; pixel < dubs.length - 4; pixel += pixelLength) {
            int res = 0;
            res |= (((int) (dubs[pixel/3] * 255.0)) & 0xff); // blue
            res |= ((((int) (dubs[(pixel/3) + 1] * 255.0)) & 0xff) << 8); // green
            res |= ((((int) (dubs[(pixel/3) + 2] * 255.0)) & 0xff) << 16); // red
            outPixels[pixel / 3] = res;
        }
        return new ImageIcon(out);
    }

    private static double[] greyToList(ImageIcon icon) {
        BufferedImage image = (BufferedImage) icon.getImage();
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        double[] res = new double[pixels.length];
        for (int i = 0; i < res.length; i++)
            res[i] = (double) (pixels[i] & 0xff) / 255.0;
        return res;
    }

    private static ImageIcon listToGrey(double[] dubs, int width, int height) {
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] outPixels = ((DataBufferInt) out.getRaster().getDataBuffer()).getData();
        for (int pixel = 0; pixel < dubs.length; pixel++) {
            int res = 0;
            res |= (((int) (dubs[pixel] * 255.0)) & 0xff); // blue
            res |= ((((int) (dubs[pixel] * 255.0)) & 0xff) << 8); // green
            res |= ((((int) (dubs[pixel] * 255.0)) & 0xff) << 16); // red
            outPixels[pixel] = res;
        }
        return new ImageIcon(out);
    }

    public static ImageIcon pathToImage(String filePath) throws IOException {
        return new ImageIcon(ImageIO.read(new File(filePath)));
    }

    public static ImageIcon listToImage(double[] dubs, int width, int height, boolean color) {
        if (color)
            return listToColor(dubs, width, height);
        else
            return listToGrey(dubs, width, height);
    }

    public static double[] imageToList(ImageIcon icon, boolean color) {
        if (color)
            return colorToList(icon);
        else
            return greyToList(icon);
    }
}
