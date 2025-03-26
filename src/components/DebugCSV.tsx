import React, { useState } from 'react';

const DebugCSV: React.FC = () => {
    const [csvContent, setCsvContent] = useState<string>('');
    const [fileStatus, setFileStatus] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const checkFile = async (filename: string) => {
        setIsLoading(true);
        setFileStatus('');
        setCsvContent('');

        try {
            const response = await fetch(`/${filename}`);
            if (response.ok) {
                const text = await response.text();
                const firstLines = text.split('\n').slice(0, 5).join('\n');
                setCsvContent(firstLines);
                setFileStatus(`File ${filename} is accessible. Showing first 5 lines.`);
            } else {
                setFileStatus(`Error accessing ${filename}: ${response.status} ${response.statusText}`);
            }
        } catch (error) {
            setFileStatus(`Failed to fetch ${filename}: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="bg-gray-800 rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold text-white mb-4">CSV File Troubleshooter</h2>

            <div className="flex space-x-4 mb-4">
                <button
                    onClick={() => checkFile('stock_data_fixed.csv')}
                    className="px-4 py-2 bg-blue-600 rounded-md text-white hover:bg-blue-700"
                    disabled={isLoading}
                >
                    Check stock_data_fixed.csv
                </button>

                <button
                    onClick={() => checkFile('stock_data_intraday.csv')}
                    className="px-4 py-2 bg-blue-600 rounded-md text-white hover:bg-blue-700"
                    disabled={isLoading}
                >
                    Check stock_data_intraday.csv
                </button>
            </div>

            {isLoading && (
                <div className="flex justify-center my-4">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
            )}

            {fileStatus && (
                <div className={`p-3 rounded-md mb-4 ${fileStatus.includes('Error') || fileStatus.includes('Failed')
                        ? 'bg-red-900/50 border border-red-800'
                        : 'bg-green-900/50 border border-green-800'
                    }`}>
                    <p className="text-white">{fileStatus}</p>
                </div>
            )}

            {csvContent && (
                <div className="bg-gray-900 p-3 rounded-md overflow-x-auto">
                    <pre className="text-gray-300 text-sm">{csvContent}</pre>
                </div>
            )}

            <div className="mt-4 text-gray-400 text-sm">
                <p>The CSV files should be located in:</p>
                <code className="bg-gray-900 px-2 py-1 rounded">trading-algo/project/public/</code>
            </div>
        </div>
    );
};

export default DebugCSV;