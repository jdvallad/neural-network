import java.awt.*;
import javax.swing.*;
import java.awt.geom.*;
import java.util.*;
public class Grapher extends JPanel{
    ArrayList<Double> coordinates;
    int mar=50;
    int max=-1;
    JFrame frame;
    protected void paintComponent(Graphics g){
        super.paintComponent(g);
        Graphics2D g1=(Graphics2D)g;
        g1.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
        int width=getWidth();
        int height=getHeight();
        g1.draw(new Line2D.Double(mar,mar,mar,height-mar));
        g1.draw(new Line2D.Double(mar,height-mar,width-mar,height-mar));
        double x=(double)(width-2*mar)/(coordinates.size()-1);
        double scale=(double)(height-2*mar)/getMax();
        g1.setPaint(Color.BLUE);
        for(int i=0;i<coordinates.size();i++){
            double x1=mar+i*x;
            double y1=height-mar-scale*coordinates.get(i);
            g1.fill(new Ellipse2D.Double(x1-2,y1-2,4,4));
        }

    }

    private double getMax(){
        if(max!=-1){
            return max;   
        }
        double m=-Integer.MAX_VALUE;
        for(int i=0;i<coordinates.size();i++){
            if(coordinates.get(i)>m){
                m=coordinates.get(i);
            }
        }
        return m;
    }       

    public Grapher(ArrayList<Double> list,int a,int b){
        coordinates=list;
        frame =new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.setSize(a,b);
        frame.setVisible(false);
    }

    public Grapher(int a,int b,int m){
        max=m;
        coordinates=new ArrayList<Double>();
        frame =new JFrame("Cost Function");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.setSize(a,b);
        frame.setVisible(false);
    }

    public Grapher(int a,int b,String pos) throws Exception{
        coordinates=new ArrayList<Double>();
        frame =new JFrame("Cost Function");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(this);
        frame.setSize(a,b);
        frame.setVisible(false);
        switch(pos){
         case "topLeftCorner":
         frame.setLocation(0,0);
         break;
         case "topRightCorner":
         frame.setLocation(1927-a,0);
         break;
         case "bottomLeftCorner":
         frame.setLocation(0,1060-b);
         break;
         case "bottomRightCorner":
         frame.setLocation(1927-a,1060-b);
         break;
         default:
         throw new Exception("Not a valid option.");
        }
    }

    public void add(double ... a){
        for(double i:a){
            coordinates.add(i);
        }
        frame.repaint();
        frame.setVisible(true);
    }
}