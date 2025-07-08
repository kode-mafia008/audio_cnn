
'use client';
import { Button } from "~/components/ui/button";
import { useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent, CardHeader } from "~/components/ui/card";

interface Prediction {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[];
}

interface VisualizationData {
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectogram: LayerData;
  waveform: WaveformData;
}

function splitLayers(visualisation: VisualizationData){
   const main: [string,LayerData][] = [];
   const internals: Record<string,[string,LayerData][]> = {};
   for (const [name,data] of Object.entries(visualisation)){
    if(!name.includes(".")){
      main.push([name,data]);
    }else{
      const [parent] = name.split('.');
      if(parent === undefined) continue;

      if(!internals[parent]) internals[parent] = [];
      internals[parent].push([name,data]); 
    }
   }
   return {main,internals};
}

export default function HomePage() {

  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [vizData, setVizData] = useState<ApiResponse | null>(null);

  const handleFileChange: any = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const baseb64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte), '')
        );

        const response = await fetch(
          'https://vmscare747--audio-cnn-inference-audioclassifier-inference.modal.run/',
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              audio_data: baseb64String,
            })
          })

        if (!response.ok) {
          throw new Error(`Network response was not ok: ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err) {
        console.error('Error processing file:', err);
        setError(err instanceof Error ? err.message : 'An unknown error occured.');
      }finally{
        setIsLoading(false);
      }
    };

    reader.onerror = (error) => {
      console.error('Error reading file:', error);
      setError('Failed to read file');
      setIsLoading(false);
    }; 
  };

 const {main,internals} = vizData ? splitLayers(vizData.visualization) : {main:[],internals:{}};

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">CNN Audio Visualisation</h1>
          <p className="mb-8 text-md text-stone-600">Upload an WAV file and visualise the audio using a CNN.</p>
        </div>
        <div className="flex flex-col items-center">
          <input
            type="file"
            accept=".wav"
            onChange={handleFileChange} 
            id="file-upload"
            className="absolute inset-0 w-full cursor-pointer backdrop-blur-sm opacity-0" />
          <Button
            variant="outline"
            size="lg"
            className="border-stone-300"
            disabled={isLoading}
          >{isLoading ? 'Analysing...' : 'Choose File'}</Button>
          {fileName && (<Badge variant="secondary" className="mt-4 bg-stone-200 text-stone-700">{fileName}</Badge>)} 
          {error && (
            <Card className="mb-8 mt-8 border-red-200 bg-red-50">
               <CardContent>
                 <p className="text-red-600">Error: {error}</p>
               </CardContent>
            </Card>
          )}
        
        {vizData && (
          <div className="space-y-8">
            <Card>
              <CardHeader>Top Predictions</CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {vizData.predictions.slice(0,3).map((prediction,index)=>(
                    <div key={index} className="space-y-2">
                      <p className="font-semibold">{prediction.class}</p>
                      <p className="text-sm text-muted-foreground">{prediction.confidence.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
        </div>
      </div>
    </main>
  );
}
