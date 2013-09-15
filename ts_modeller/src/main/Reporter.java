/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package main;

import java.io.File;
import java.io.FileWriter;
import java.util.LinkedList;

/**
 *
 * @author stuart
 */
public class Reporter {
    
    int epochs;
    String dataFile;
    LinkedList<Datum> data;
    
    Reporter(int iter, String file) {
        epochs = iter;
        data = new LinkedList();
        for (int i = 0; i < epochs; i++) {
            data.add(new Datum(i));
        }
        dataFile = file;
    }
    
    public void add(Datum dat) {
        data.add(dat);
    }
    
    public Datum get(int epoch) {
        return data.get(epoch);
    }
    
    public void printData(String fileName) {
        try {
            String fullName = fileName;
            File f = new File(fullName);
            FileWriter fwriter = new FileWriter(fullName);
            
            String lineOut = "Iteration,AvgElmanSSE,AvgElmanPE,AvgJordanSSE,AvgJordanPE,"
                    + "AvgFFNNSSE,AvgFFNNPE,AvgStandardPSOSSE,AvgStandardPE,AvgChargedPSOSSE,AvgChargedPE\n";
            fwriter.write(lineOut);
            
            for (Datum d : data) {
                lineOut = "" + d.getIteration() + ",";
                lineOut += d.getAvgElmanSSE() + ",";
                lineOut += d.getAvgElmanPE() + ",";
                lineOut += d.getAvgJordanSSE() + ",";
                lineOut += d.getAvgJordanPE() + ",";
                lineOut += d.getAvgFFNNSSE() + ",";
                lineOut += d.getAvgFFNNPE() + ",";
                lineOut += d.getAvgStandardPSOSSE() + ",";
                lineOut += d.getAvgStandardPSOPE() + ",";
                lineOut += d.getAvgChargedPSOSSE() + ",";
                lineOut += d.getAvgChargedPSOPE() + "\n";
                fwriter.write(lineOut);
                fwriter.flush();
            }
        } catch (Exception err) {
            err.printStackTrace();
        }
    }
}
