"use client"

import { useRef, useState, useMemo, useEffect, type RefObject, type MutableRefObject } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import * as THREE from "three"

// Video auto-discovery
// Add files named video1.mp4 ... video30.mp4 in /public/videos/
// Optional no-audio files: video1noaudio.mp4 in /public/videos-no-audio/
// They will be detected automatically at runtime.

type VideoItem = {
  id: number
  title: string
  videoUrl: string
}

const TILE_WIDTH = 1.3
const TILE_HEIGHT = 0.82

const createCurvedPlane = (width: number, height: number, radius: number) => {
  const safeRadius = Math.max(radius, 0.01)
  const geo = new THREE.PlaneGeometry(width, height, 16, 10)
  const position = geo.attributes.position as THREE.BufferAttribute

  for (let i = 0; i < position.count; i += 1) {
    const x = position.getX(i)
    const y = position.getY(i)
    const inside = safeRadius * safeRadius - x * x - y * y
    const z = Math.sqrt(Math.max(inside, 0)) - safeRadius
    position.setZ(i, z)
  }

  position.needsUpdate = true
  geo.computeVertexNormals()
  return geo
}

type HandState = {
  x: number
  y: number
  pinch: boolean
  scale: number
  visible: boolean
  landmarks: { x: number; y: number }[] | null
}

type HandTrackerStatus = "idle" | "initializing" | "running" | "error"

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17]
]

function useHandTracking(videoRef: RefObject<HTMLVideoElement>) {
  const handRef = useRef<HandState>({
    x: 0.5,
    y: 0.5,
    pinch: false,
    scale: 0,
    visible: false,
    landmarks: null
  })
  const [status, setStatus] = useState<HandTrackerStatus>("idle")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    let stream: MediaStream | null = null
    let landmarker: { detectForVideo?: (video: HTMLVideoElement, timestamp: number) => any; close?: () => void } | null = null
    let rafId = 0
    let lastVideoTime = -1

    const start = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("error")
        setError("Camera access is not supported in this browser.")
        return
      }

      setStatus("initializing")

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        })

        if (!videoRef.current) return
        videoRef.current.srcObject = stream
        await videoRef.current.play()

        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision")
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        )
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        })

        if (!active) return
        setStatus("running")

        const loop = () => {
          if (!active || !videoRef.current || !landmarker?.detectForVideo) return
          const video = videoRef.current
          if (video.readyState < 2) {
            rafId = requestAnimationFrame(loop)
            return
          }

          if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime
            const results = landmarker.detectForVideo(video, performance.now())
            const landmarks = results?.landmarks?.[0]
            if (landmarks && landmarks.length > 12) {
              const indexTip = landmarks[8]
              const thumbTip = landmarks[4]
              const wrist = landmarks[0]
              const middleTip = landmarks[12]
              const pinchDistance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y)
              const handScale = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y)
              const pinch = pinchDistance < 0.065
              const mirroredX = 1 - indexTip.x
              const rawLandmarks = landmarks.map((point: { x: number; y: number }) => ({
                x: point.x,
                y: point.y
              }))
              handRef.current = {
                x: mirroredX,
                y: indexTip.y,
                pinch,
                scale: handScale,
                visible: true,
                landmarks: rawLandmarks
              }
            } else {
              handRef.current = {
                ...handRef.current,
                pinch: false,
                scale: 0,
                visible: false,
                landmarks: null
              }
            }
          }

          rafId = requestAnimationFrame(loop)
        }

        rafId = requestAnimationFrame(loop)
      } catch (err) {
        if (!active) return
        setStatus("error")
        setError(err instanceof Error ? err.message : "Camera access failed.")
      }
    }

    start()

    return () => {
      active = false
      if (rafId) cancelAnimationFrame(rafId)
      if (stream) stream.getTracks().forEach((track) => track.stop())
      if (landmarker?.close) landmarker.close()
    }
  }, [videoRef])

  return { handRef, status, error }
}

interface VideoTileProps {
  position: [number, number, number]
  quaternion: THREE.Quaternion
  video: VideoItem
  onVideoMissing: (videoUrl: string) => void
  audioEnabled: boolean
  handRef?: MutableRefObject<HandState>
  sphereRadius: number
}

function VideoTile({
  position,
  quaternion,
  video,
  onVideoMissing,
  audioEnabled,
  handRef,
  sphereRadius
}: VideoTileProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoTexture, setVideoTexture] = useState<THREE.VideoTexture | null>(null)
  const { camera } = useThree()
  const isNoAudioPreview = useMemo(
    () => /noaudio/i.test(video.videoUrl) || video.videoUrl.startsWith("/videos-no-audio/"),
    [video.videoUrl]
  )
  const volumeRef = useRef(0)
  const muffleRef = useRef(0)
  const worldPos = useMemo(() => new THREE.Vector3(), [])
  const toTile = useMemo(() => new THREE.Vector3(), [])
  const cameraDir = useMemo(() => new THREE.Vector3(), [])
  const geometry = useMemo(
    () => createCurvedPlane(TILE_WIDTH, TILE_HEIGHT, sphereRadius),
    [sphereRadius]
  )
  const hoverGeometry = useMemo(
    () => createCurvedPlane(TILE_WIDTH * 1.06, TILE_HEIGHT * 1.1, sphereRadius),
    [sphereRadius]
  )

  useEffect(() => {
    // Create video element for texture
    const videoElement = document.createElement("video")
    videoElement.src = video.videoUrl
    videoElement.crossOrigin = "anonymous"
    videoElement.loop = true
    videoElement.muted = true
    videoElement.volume = 1
    videoElement.playsInline = true
    videoElement.autoplay = true
    
    const handleLoaded = () => {
      const texture = new THREE.VideoTexture(videoElement)
      texture.colorSpace = THREE.SRGBColorSpace
      texture.minFilter = THREE.LinearFilter
      texture.magFilter = THREE.LinearFilter
      texture.wrapS = THREE.RepeatWrapping
      texture.repeat.x = -1
      texture.offset.x = 1
      setVideoTexture(texture)
      videoElement.play().catch(() => {
        // Autoplay might be blocked, that's okay
      })
    }

    const handleError = () => {
      onVideoMissing(video.videoUrl)
    }

    videoElement.addEventListener("loadeddata", handleLoaded)
    videoElement.addEventListener("error", handleError)

    videoRef.current = videoElement

    return () => {
      videoElement.removeEventListener("loadeddata", handleLoaded)
      videoElement.removeEventListener("error", handleError)
      videoElement.pause()
      videoElement.src = ""
      if (videoTexture) {
        videoTexture.dispose()
      }
    }
  }, [video.videoUrl, onVideoMissing])
  
  useEffect(() => {
    const videoElement = videoRef.current
    if (!videoElement) return
    videoElement.muted = isNoAudioPreview ? true : !audioEnabled
    if (audioEnabled && !isNoAudioPreview) {
      videoElement.play().catch(() => {
        // Autoplay with audio might be blocked until a user gesture
      })
    }
  }, [audioEnabled, isNoAudioPreview])

  useFrame((_, delta) => {
    const videoElement = videoRef.current
    if (!videoElement) return

    if (!audioEnabled || isNoAudioPreview) {
      if (!videoElement.muted) videoElement.muted = true
      if (videoElement.volume !== 0) videoElement.volume = 0
      volumeRef.current = 0
      muffleRef.current = 0
      return
    }

    if (videoElement.muted) videoElement.muted = false

    if (meshRef.current) {
      meshRef.current.getWorldPosition(worldPos)
      toTile.copy(worldPos).sub(camera.position).normalize()
      camera.getWorldDirection(cameraDir)
      const dot = cameraDir.dot(toTile)
      const target = THREE.MathUtils.smoothstep(dot, 0.2, 0.95)
      const pinchTarget = handRef?.current?.pinch ? 1 : 0
      muffleRef.current = THREE.MathUtils.damp(muffleRef.current, pinchTarget, 6, delta)
      const muffle = THREE.MathUtils.lerp(1, 0.25, muffleRef.current)
      const next = THREE.MathUtils.damp(volumeRef.current, target * muffle, 6, delta)
      volumeRef.current = next
      videoElement.volume = next
    }
  })

  return (
    <mesh
      ref={meshRef}
      position={position}
      quaternion={quaternion}
      userData={{ video }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      geometry={geometry}
    >
      {videoTexture ? (
        <meshBasicMaterial
          map={videoTexture}
          side={THREE.DoubleSide}
          toneMapped={false}
        />
      ) : (
        <meshStandardMaterial
          color={`hsl(${video.id * 30}, 70%, 50%)`}
          side={THREE.DoubleSide}
        />
      )}
      {hovered && (
        <mesh position={[0, 0, 0.02]} geometry={hoverGeometry}>
          <meshBasicMaterial color="#ffffff" opacity={0.2} transparent side={THREE.DoubleSide} />
        </mesh>
      )}
    </mesh>
  )
}

interface OrbProps {
  audioEnabled: boolean
  handRef?: MutableRefObject<HandState>
  videos: VideoItem[]
  onVideoMissing: (videoUrl: string) => void
}

function Orb({ audioEnabled, handRef, videos, onVideoMissing }: OrbProps) {
  const groupRef = useRef<THREE.Group>(null)
  const { camera } = useThree()
  const smoothScaleRef = useRef(0)
  const orbScaleRef = useRef(1)
  const minZoom = 3
  const maxZoom = 15
  const minScale = 0.08
  const maxScale = 0.32
  const minOrbScale = 0.05
  const maxOrbScale = 1
  
  // Generate packed positions on a sphere for uploaded video tiles
  const { positions: tilePositions, radius: sphereRadius } = useMemo(() => {
    const positions: { position: [number, number, number]; quaternion: THREE.Quaternion }[] = []
    const count = videos.length
    if (count === 0) return { positions, radius: 1 }

    const density = 0.7
    const minRadius = 2.6
    const maxRadius = 6.8
    const gapFactor = 1.2
    let radius = Math.sqrt((count * TILE_WIDTH * TILE_HEIGHT) / (4 * Math.PI * density))
    radius = Math.min(Math.max(radius, minRadius), maxRadius)

    const countSlots = (r: number) => {
      const latStep = (TILE_HEIGHT * gapFactor) / r
      const rows = Math.max(1, Math.floor(Math.PI / latStep))
      let total = 0
      for (let row = 0; row < rows; row += 1) {
        const lat = -Math.PI / 2 + (row + 0.5) * (Math.PI / rows)
        const rowRadius = r * Math.cos(lat)
        const tilesInRow = Math.max(
          1,
          Math.floor((2 * Math.PI * rowRadius) / (TILE_WIDTH * gapFactor))
        )
        total += tilesInRow
      }
      return total
    }

    let bestRadius = radius
    let bestDiff = Number.POSITIVE_INFINITY
    for (let i = 0; i < 20; i += 1) {
      const slots = countSlots(radius)
      const diff = Math.abs(slots - count)
      if (diff < bestDiff) {
        bestDiff = diff
        bestRadius = radius
      }
      radius = slots < count ? radius * 1.08 : radius * 0.94
      radius = Math.min(Math.max(radius, minRadius), maxRadius)
    }

    radius = bestRadius
    const latStep = (TILE_HEIGHT * gapFactor) / radius
    const rows = Math.max(1, Math.floor(Math.PI / latStep))
    const packed: typeof positions = []
    const up = new THREE.Vector3(0, 1, 0)
    const fallbackUp = new THREE.Vector3(0, 0, 1)
    const normal = new THREE.Vector3()
    const yAxis = new THREE.Vector3()
    const xAxis = new THREE.Vector3()
    const basis = new THREE.Matrix4()

    for (let row = 0; row < rows; row += 1) {
      const lat = -Math.PI / 2 + (row + 0.5) * (Math.PI / rows)
      const rowRadius = radius * Math.cos(lat)
      const tilesInRow = Math.max(
        1,
        Math.floor((2 * Math.PI * rowRadius) / (TILE_WIDTH * gapFactor))
      )
      for (let col = 0; col < tilesInRow; col += 1) {
        const theta = (2 * Math.PI * col) / tilesInRow
        const x = rowRadius * Math.cos(theta)
        const z = rowRadius * Math.sin(theta)
        const y = radius * Math.sin(lat)
        normal.set(x, y, z).normalize()
        yAxis.copy(up).projectOnPlane(normal)
        if (yAxis.lengthSq() < 1e-6) {
          yAxis.copy(fallbackUp).projectOnPlane(normal)
        }
        yAxis.normalize()
        xAxis.crossVectors(yAxis, normal).normalize()
        basis.makeBasis(xAxis, yAxis, normal)
        const quaternion = new THREE.Quaternion().setFromRotationMatrix(basis)

        packed.push({
          position: [x, y, z],
          quaternion
        })
      }
    }

    if (packed.length <= count) return { positions: packed, radius }

    const step = packed.length / count
    for (let i = 0; i < count; i += 1) {
      positions.push(packed[Math.floor(i * step)])
    }

    return { positions, radius }
  }, [videos.length])

  // Handle keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!groupRef.current) return
      
      const rotationSpeed = 0.1
      const zoomSpeed = 0.5
      
      switch (e.key) {
        case "ArrowLeft":
          groupRef.current.rotation.y -= rotationSpeed
          break
        case "ArrowRight":
          groupRef.current.rotation.y += rotationSpeed
          break
        case "ArrowUp":
          groupRef.current.rotation.x -= rotationSpeed
          break
        case "ArrowDown":
          groupRef.current.rotation.x += rotationSpeed
          break
        case "+":
        case "=":
          camera.position.z = Math.max(camera.position.z - zoomSpeed, 3)
          break
        case "-":
        case "_":
          camera.position.z = Math.min(camera.position.z + zoomSpeed, 15)
          break
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [camera])

  // Auto-rotate slowly, or follow the hand when tracked
  useFrame((_, delta) => {
    if (groupRef.current) {
      const hand = handRef?.current
      if (hand?.visible) {
        const targetY = (hand.x - 0.5) * Math.PI * 2.2
        const targetX = (0.5 - hand.y) * Math.PI * 0.6
        groupRef.current.rotation.y = THREE.MathUtils.damp(
          groupRef.current.rotation.y,
          targetY,
          8,
          delta
        )
        groupRef.current.rotation.x = THREE.MathUtils.damp(
          groupRef.current.rotation.x,
          targetX,
          8,
          delta
        )

        smoothScaleRef.current = THREE.MathUtils.damp(smoothScaleRef.current, hand.scale, 10, delta)
        const zoomT = THREE.MathUtils.clamp(
          (smoothScaleRef.current - minScale) / (maxScale - minScale),
          0,
          1
        )
        const targetZ = THREE.MathUtils.lerp(maxZoom, minZoom, zoomT)
        camera.position.z = THREE.MathUtils.damp(camera.position.z, targetZ, 8, delta)

        const spreadT = THREE.MathUtils.clamp(zoomT * 1.15, 0, 1)
        const targetOrbScale = hand.pinch
          ? minOrbScale
          : THREE.MathUtils.lerp(0.55, maxOrbScale, spreadT)
        orbScaleRef.current = THREE.MathUtils.damp(orbScaleRef.current, targetOrbScale, 8, delta)
        groupRef.current.scale.setScalar(orbScaleRef.current)

      } else {
        groupRef.current.rotation.y += delta * 0.1
        smoothScaleRef.current = 0
        orbScaleRef.current = THREE.MathUtils.damp(orbScaleRef.current, maxOrbScale, 6, delta)
        groupRef.current.scale.setScalar(orbScaleRef.current)
      }
    }
  })

  return (
    <group ref={groupRef}>
      {tilePositions.map((tile, index) => {
        return (
          <group key={index}>
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  array={
                    new Float32Array([
                      0,
                      0,
                      0,
                      tile.position[0],
                      tile.position[1],
                      tile.position[2]
                    ])
                  }
                  count={2}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial
                color={`hsl(${videos[index].id * 30}, 80%, 60%)`}
                transparent
                opacity={0.7}
              />
            </line>
            <VideoTile
              position={tile.position}
              quaternion={tile.quaternion}
              video={videos[index]}
              onVideoMissing={onVideoMissing}
              audioEnabled={audioEnabled}
              handRef={handRef}
              sphereRadius={sphereRadius}
            />
          </group>
        )
      })}
    </group>
  )
}

export default function VideoOrb() {
  const [videos, setVideos] = useState<VideoItem[]>([])
  const [audioEnabled, setAudioEnabled] = useState(false)
  const [muted, setMuted] = useState(false)
  const handVideoRef = useRef<HTMLVideoElement>(null)
  const handCanvasRef = useRef<HTMLCanvasElement>(null)
  const { handRef, status: handStatus, error: handError } = useHandTracking(handVideoRef)
  const maxVideos = 30

  useEffect(() => {
    let active = true

    const shuffle = (items: VideoItem[]) => {
      const next = [...items]
      for (let i = next.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[next[i], next[j]] = [next[j], next[i]]
      }
      return next
    }

    const checkUrl = async (url: string) => {
      try {
        const head = await fetch(url, { method: "HEAD" })
        if (head.ok) return true
        if (head.status === 405) {
          const get = await fetch(url)
          return get.ok
        }
        return false
      } catch {
        return false
      }
    }

    const loadVideos = async () => {
      const mainUrls = Array.from({ length: maxVideos }, (_, i) => `/videos/video${i + 1}.mp4`)
      const noAudioUrls = Array.from(
        { length: maxVideos },
        (_, i) => `/videos-no-audio/video${i + 1}noaudio.mp4`
      )
      const noAudioAltUrls = Array.from(
        { length: maxVideos },
        (_, i) => `/videos-no-audio/video${i + 1}noaudio.mp4.mp4`
      )

      const [mainChecks, noAudioChecks, noAudioAltChecks] = await Promise.all([
        Promise.all(mainUrls.map(checkUrl)),
        Promise.all(noAudioUrls.map(checkUrl)),
        Promise.all(noAudioAltUrls.map(checkUrl))
      ])

      const found: VideoItem[] = []
      let id = 1

      for (let i = 0; i < maxVideos; i += 1) {
        if (mainChecks[i]) {
          found.push({
            id,
            title: `Video ${i + 1}`,
            videoUrl: mainUrls[i]
          })
          id += 1
        }
      }

      for (let i = 0; i < maxVideos; i += 1) {
        const url = noAudioChecks[i] ? noAudioUrls[i] : noAudioAltChecks[i] ? noAudioAltUrls[i] : null
        if (url) {
          found.push({
            id,
            title: `No Audio ${i + 1}`,
            videoUrl: url
          })
          id += 1
        }
      }

      if (active) {
        setVideos(shuffle(found))
      }
    }

    loadVideos()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (audioEnabled || muted) return
    const enableAudio = () => setAudioEnabled(true)
    window.addEventListener("pointerdown", enableAudio, { once: true })
    window.addEventListener("keydown", enableAudio, { once: true })
    return () => {
      window.removeEventListener("pointerdown", enableAudio)
      window.removeEventListener("keydown", enableAudio)
    }
  }, [audioEnabled])

  const handleVideoMissing = (videoUrl: string) => {
    setVideos((prev) => prev.filter((video) => video.videoUrl !== videoUrl))
  }

  useEffect(() => {
    let rafId = 0
    const draw = () => {
      const canvas = handCanvasRef.current
      const video = handVideoRef.current
      if (!canvas || !video) {
        rafId = requestAnimationFrame(draw)
        return
      }

      const videoWidth = video.videoWidth || 640
      const videoHeight = video.videoHeight || 480
      const dpr = window.devicePixelRatio || 1

      if (canvas.width !== videoWidth * dpr || canvas.height !== videoHeight * dpr) {
        canvas.width = videoWidth * dpr
        canvas.height = videoHeight * dpr
      }

      const ctx = canvas.getContext("2d")
      if (!ctx) {
        rafId = requestAnimationFrame(draw)
        return
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, videoWidth, videoHeight)

      const hand = handRef.current
      const landmarks = hand.landmarks
      if (hand.visible && landmarks && landmarks.length > 0) {
        ctx.lineWidth = 2
        ctx.strokeStyle = "rgba(59, 130, 246, 0.9)"
        ctx.fillStyle = "rgba(14, 165, 233, 0.9)"

        ctx.beginPath()
        for (const [start, end] of HAND_CONNECTIONS) {
          const a = landmarks[start]
          const b = landmarks[end]
          if (!a || !b) continue
          ctx.moveTo(a.x * videoWidth, a.y * videoHeight)
          ctx.lineTo(b.x * videoWidth, b.y * videoHeight)
        }
        ctx.stroke()

        for (const point of landmarks) {
          const x = point.x * videoWidth
          const y = point.y * videoHeight
          ctx.beginPath()
          ctx.arc(x, y, 3, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      rafId = requestAnimationFrame(draw)
    }

    rafId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafId)
  }, [handRef])

  return (
    <div className="relative w-full h-screen bg-transparent">
      {/* Instructions */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-10 text-center">
        <h1 className="text-2xl md:text-3xl font-semibold text-slate-100 mb-2">Moments, Suspended</h1>
        <p className="text-slate-300 text-sm md:text-base">Tap to enter. Pinch to gather, open your hand to release.</p>
      </div>

      <button
        type="button"
        onClick={() => {
          setMuted((prev) => !prev)
          if (!audioEnabled) setAudioEnabled(true)
        }}
        className="absolute top-5 left-5 z-20 rounded-full border border-white/20 bg-black/40 px-4 py-2 text-xs uppercase tracking-[0.2em] text-white backdrop-blur transition hover:bg-black/60"
        aria-pressed={muted}
        aria-label={muted ? "Unmute" : "Mute"}
      >
        {muted ? "Muted" : "Mute"}
      </button>

      {/* Camera preview */}
      <div className="absolute top-4 right-4 z-20 w-56 md:w-64">
        <div className="rounded-xl border border-slate-200 bg-white/90 shadow-lg backdrop-blur">
          <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-slate-500">
            Hand Camera
          </div>
          <div className="relative aspect-video overflow-hidden rounded-b-xl bg-slate-900">
            <video
              ref={handVideoRef}
              className="h-full w-full object-cover -scale-x-100"
              muted
              playsInline
              autoPlay
            />
            <canvas
              ref={handCanvasRef}
              className="absolute inset-0 h-full w-full pointer-events-none -scale-x-100"
            />
            <div className="absolute bottom-1 right-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
              {handStatus === "running"
                ? "Tracking"
                : handStatus === "initializing"
                  ? "Starting"
                  : handStatus === "error"
                    ? "Camera Off"
                    : "Idle"}
            </div>
          </div>
          {handError && (
            <div className="px-2 py-2 text-[11px] text-rose-600">{handError}</div>
          )}
        </div>
      </div>

      <Canvas
        camera={{ position: [0, 0, 10], fov: 50 }}
        style={{ cursor: "grab", background: "transparent" }}
      >
        <ambientLight intensity={1} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.4} />
        
        <Orb
          audioEnabled={audioEnabled && !muted}
          handRef={handRef}
          videos={videos}
          onVideoMissing={handleVideoMissing}
        />
        
        <OrbitControls
          enablePan={false}
          minDistance={3}
          maxDistance={15}
          enableDamping
          dampingFactor={0.05}
        />
      </Canvas>

    </div>
  )
}
