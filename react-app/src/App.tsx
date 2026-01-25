import { useState, useEffect } from 'react'
import './App.css'

// Define neuron interface
interface Neuron {
  id: number
  value: number
  layer: number
  distance: number
  weights: Record<number, number>
}

interface Snapshot {
  epoch: number
  neurons: Neuron[]
}

/*
* Render a radial tree diagram of neurons
*/
function renderTreeDiagram(neurons: Neuron[], width = 800, height = 800) {
  const layers: Record<number, Neuron[]> = {}

  // Group neurons by layer
  neurons.forEach(n => {
    if (!layers[n.layer]) layers[n.layer] = []
    layers[n.layer].push(n)
  })

  // Convert layers object to sorted array
  const layerArray: Neuron[][] = Object.keys(layers)
    .map(Number)
    .sort((a, b) => a - b)
    .map(layer => layers[layer])

  // Store positions for each neuron
  const tree_map: [number, number, Neuron][][] = []

  const scale = 200 // scale up the positions for visibility

  layerArray.forEach((layer, layerIndex) => {
    const layer_tree_map: [number, number, Neuron][] = []
    const length = layer.length

    layer.forEach((i, index) => {
      const distance = i.layer === 0 ? 1 : Number(i.distance.toFixed(2)) + 1

      // Offset for alternating layers
      const offset = (layerIndex % 2 === 0 ? 0 : Math.PI / length)
      const rads = (2 * Math.PI * index) / length + offset

      const x = Number((Math.sin(rads) * distance * scale).toFixed(2))
      const y = Number((Math.cos(rads) * distance * scale).toFixed(2))

      layer_tree_map.push([x, y, i])
    })

    tree_map.push(layer_tree_map)
  })

  const cx = width / 2
  const cy = height / 2

  return (
    <svg width={width} height={height} style={{ border: '1px solid #ccc' }}>
      {/* Draw connections between neurons */}
      {tree_map.map((layer, layerIndex) => {
        if (layerIndex === 0) return null
        const prevLayer = tree_map[layerIndex - 1]
        return layer.flatMap(([x, y, neuron]) =>
          prevLayer.map(([px, py, prevNeuron]) => {
            const weight = neuron.weights[prevNeuron.id] ?? 0.1
            const strokeOpacity = Math.min(Math.max(weight, 0.1), 1)
            return (
              // Draw lines between neurons
              <line
                key={`${prevNeuron.id}-${neuron.id}`}
                x1={cx + px}
                y1={cy - py}
                x2={cx + x}
                y2={cy - y}
                stroke="black"
                strokeWidth={2}
                strokeOpacity={strokeOpacity}
              />
            )
          })
        )
      })}

      {/* Draw neurons as colored circles */}
      {tree_map.map(layer =>
        layer.map(([x, y, neuron]) => {
          const hue = (neuron.layer * 60) % 360
          const lightness = 30 + neuron.value * 50
          const fill = `hsl(${hue}, 80%, ${lightness}%)`
          return (
            <circle
              key={neuron.id}
              cx={cx + x}
              cy={cy - y}
              r={6}
              fill={fill}
              stroke="black"
            />
          )
        })
      )}
    </svg>
  )
}

function App() {
  const divtitle = 'Tree Diagram'
  const [epochIndex, setEpochIndex] = useState(0)
  const [allData, setAllData] = useState<Snapshot[]>([])

  useEffect(() => {
    // Fetch list of snapshot JSON files from public/snapshots
    fetch('/snapshots/index.json')
      .then(res => res.json())
      .then((files: string[]) => {
        // Fetch each snapshot JSON file
        return Promise.all(
          files.map(f =>
            fetch(`/snapshots/${f}`).then(res => res.json())
          )
        )
      })
      .then(data => setAllData(data))
      .catch(err => console.error('Failed to load snapshots:', err))
  }, [])

  console.log('Epoch data:', allData[epochIndex])
  return (
    <div style={{ padding: '20px' }}>
      {/* Title */}
      <div className="my-div" style={{ fontSize: '24px', marginBottom: '10px' }}>
        {divtitle}
      </div>

      {/* Navigation buttons: Prev and Next */}
      <div style={{ marginBottom: '20px' }}>
        <button onClick={() => setEpochIndex(i => Math.max(i - 1, 0))}>Prev</button>
        <button
          onClick={() =>
            setEpochIndex(i => Math.min(i + 1, allData.length - 1))
          }
          style={{ marginLeft: '10px' }}
        >
          Next
        </button>
      </div>

      {/* Render tree diagram or loading */}
      {allData[epochIndex]?.neurons ? (
        // if the array contains neurons
        renderTreeDiagram(allData[epochIndex].neurons, 800, 800)
      ) : (
        // else
        <p>Loading snapshots...</p>
      )}
    </div>
)}

export default App