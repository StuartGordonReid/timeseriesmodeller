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

    String dataFile;
    LinkedList<Datum> data;
    int epochs;

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

    public void printToCSV(String fileName) {
        try {
            String fullName = fileName;
            File f = new File(fullName);
            FileWriter fwriter = new FileWriter(fullName);

            String lineOut = "Iteration,ElmanSSE,ElmanPE,JordanSSE,JordanPE,"
                    + "FFNNSSE,FFNNPE,QuantumPSOSSE,QuantumPSOPE,ChargedPSOSSE,"
                    + "ChargedPSOPE,StandardPSOSSE,StandardPE,Goal";
            fwriter.write(lineOut);

            for (Datum d : data) {
                lineOut = "" + d.getIteration() + ",";
                lineOut += d.getElmanSSE() + ",";
                lineOut += d.getElmanPE() + ",";
                lineOut += d.getJordanSSE() + ",";
                lineOut += d.getJordanPE() + ",";
                lineOut += d.getFFNNSSE() + ",";
                lineOut += d.getFFNNPE() + ",";
                lineOut += d.getQuantumPSOSSE() + ",";
                lineOut += d.getQuantumPSOPE() + ",";
                lineOut += d.getChargedPSOSSE() + ",";
                lineOut += d.getChargedPSOPE() + ",";
                lineOut += d.getStandardPSOSSE() + ",";
                lineOut += d.getStandardPSOPE() + ",";
                lineOut += d.getGoal() + "\n";
                fwriter.write(lineOut);
                fwriter.flush();
            }
        } catch (Exception err) {
            err.printStackTrace();
        }
    }

}
