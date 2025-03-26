import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';

const TestChart: React.FC = () => {
    const chartRef = useRef<HTMLDivElement>(null);
    // Use a ref to track if the component is mounted
    const isMountedRef = useRef(true);
    const [error, setError] = useState<string | null>(null);
    const chartInstanceRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        // Set mounted flag
        isMountedRef.current = true;
        
        // Define chart creation with delay
        const createChartWithDelay = () => {
            if (!isMountedRef.current) return; // Skip if unmounted
            
            try {
                if (!chartRef.current) {
                    console.log("Chart container not ready");
                    return;
                }
                
                // Create new chart
                console.log("Creating basic chart...");
                
                const chart = createChart(chartRef.current, {
                    width: chartRef.current.clientWidth || 600,
                    height: 300,
                    layout: {
                        background: { color: '#1f2937' },
                        textColor: '#d1d5db',
                    },
                    grid: {
                        vertLines: { color: '#2a2e39' },
                        horzLines: { color: '#2a2e39' },
                    }
                });
                
                // Save reference ONLY if still mounted
                if (isMountedRef.current) {
                    chartInstanceRef.current = chart;
                    
                    // Add a line series with dummy data
                    const lineSeries = chart.addLineSeries({
                        color: '#2962FF',
                        lineWidth: 2,
                    });
                    
                    // Use data that's guaranteed to work
                    lineSeries.setData([
                        { time: '2022-01-01', value: 10 },
                        { time: '2022-01-02', value: 12 },
                        { time: '2022-01-03', value: 8 },
                        { time: '2022-01-04', value: 15 },
                        { time: '2022-01-05', value: 13 }
                    ]);
                    
                    chart.timeScale().fitContent();
                    setError(null);
                } else {
                    // If component unmounted during creation, remove the chart
                    chart.remove();
                }
            } catch (err) {
                console.error("Error creating chart:", err);
                if (isMountedRef.current) {
                    setError(`Failed to create basic chart: ${err instanceof Error ? err.message : String(err)}`);
                }
            }
        };

        // Clean up previous chart instance if it exists
        if (chartInstanceRef.current) {
            try {
                chartInstanceRef.current.remove();
            } catch (e) {
                console.warn("Error removing existing chart:", e);
            }
            chartInstanceRef.current = null;
        }
        
        // Create chart with a delay
        const timeoutId = setTimeout(createChartWithDelay, 100);
        
        // Cleanup function
        return () => {
            // Mark component as unmounted
            isMountedRef.current = false;
            
            // Clear timeout
            clearTimeout(timeoutId);
            
            // Clean up chart instance
            if (chartInstanceRef.current) {
                try {
                    chartInstanceRef.current.remove();
                } catch (e) {
                    console.warn("Error during cleanup:", e);
                }
                chartInstanceRef.current = null;
            }
        };
    }, []);

    return (
        <div className="p-4 bg-gray-800 rounded-lg">
            <h2 className="text-lg font-semibold mb-4 text-white">Basic Chart Demo</h2>
            
            {error && (
                <div className="p-3 mb-4 bg-red-900/50 border border-red-800 rounded text-red-200">
                    {error}
                </div>
            )}
            
            <div 
                ref={chartRef} 
                className="w-full h-[300px] bg-gray-900/50 rounded"
                style={{ minHeight: "300px" }}
            />
        </div>
    );
};

export default TestChart;