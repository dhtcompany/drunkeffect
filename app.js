(function () {
  "use strict";

  const MAX_GHOST_FRAMES = 4;
  const MOTION_SAMPLE_WIDTH = 24;
  const MOTION_SAMPLE_HEIGHT = 14;
  const GHOST_WEIGHTS = [0.14, 0.2, 0.28, 0.36];
  const GHOST_CAPTURE_INTERVAL_MS = 180;
  const TEMPORAL_LAG_BLEND = 0.14;

  function getCoverCrop(sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const sourceAspect = sourceWidth / sourceHeight;
    const targetAspect = targetWidth / targetHeight;

    let sx = 0;
    let sy = 0;
    let sWidth = sourceWidth;
    let sHeight = sourceHeight;

    if (sourceAspect > targetAspect) {
      sWidth = sourceHeight * targetAspect;
      sx = (sourceWidth - sWidth) * 0.5;
    } else {
      sHeight = sourceWidth / targetAspect;
      sy = (sourceHeight - sHeight) * 0.5;
    }

    return { sx, sy, sWidth, sHeight };
  }

  function create2DContext(canvas, options) {
    return canvas.getContext("2d", options) || canvas.getContext("2d");
  }

  function clamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function lerp(start, end, amount) {
    return start + ((end - start) * amount);
  }

  function getEffectLevel(intensity) {
    return Math.pow(clamp01(intensity), 1.65);
  }

  function getSway(timeSeconds, intensity, motionAmount) {
    const base =
      lerp(0.05, 2.37, intensity) +
      (motionAmount * lerp(0.1, 0.75, intensity));

    return {
      x:
        (Math.sin(timeSeconds * 0.62) * 0.65 +
          Math.cos(timeSeconds * 0.27) * 0.35 +
          Math.sin(timeSeconds * 1.14) * 0.18) * base,
      y:
        (Math.cos(timeSeconds * 0.51) * 0.52 +
          Math.sin(timeSeconds * 0.31) * 0.26 +
          Math.cos(timeSeconds * 0.93) * 0.14) * base,
    };
  }

  function getWobbleJitter(timeSeconds, intensity) {
    const amount = lerp(0.05, 2.05, intensity);

    return {
      x:
        (Math.sin(timeSeconds * 2.9) * 0.45 +
          Math.cos(timeSeconds * 4.2) * 0.2 +
          Math.sin(timeSeconds * 7.8) * 0.12) * amount,
      y:
        (Math.cos(timeSeconds * 3.4) * 0.4 +
          Math.sin(timeSeconds * 5.3) * 0.18 +
          Math.cos(timeSeconds * 8.6) * 0.1) * amount,
      rotation:
        (Math.sin(timeSeconds * 0.82) * 0.009 +
          Math.cos(timeSeconds * 0.47) * 0.006 +
          Math.sin(timeSeconds * 2.3) * 0.0035) * (0.6 + intensity),
    };
  }

  function getPinchAmount(timeSeconds, intensity) {
    return Math.sin((timeSeconds / 30) * Math.PI) * lerp(0.015, 0.2, intensity);
  }

  class MotionEstimator {
    constructor() {
      this.canvas = document.createElement("canvas");
      this.canvas.width = MOTION_SAMPLE_WIDTH;
      this.canvas.height = MOTION_SAMPLE_HEIGHT;
      this.ctx = create2DContext(this.canvas, { willReadFrequently: true });
      this.prev = new Uint8Array(MOTION_SAMPLE_WIDTH * MOTION_SAMPLE_HEIGHT);
      this.hasPrev = false;
      this.frameIndex = 0;
      this.motion = 0;
      this.smoothed = 0;
    }

    sample(video) {
      this.frameIndex += 1;

      if (!this.ctx || this.frameIndex % 3 !== 0) {
        return this.smoothed;
      }

      this.ctx.drawImage(video, 0, 0, MOTION_SAMPLE_WIDTH, MOTION_SAMPLE_HEIGHT);
      const imageData = this.ctx.getImageData(
        0,
        0,
        MOTION_SAMPLE_WIDTH,
        MOTION_SAMPLE_HEIGHT,
      ).data;

      let diffSum = 0;
      let pixelCount = 0;

      for (let index = 0, pixel = 0; index < imageData.length; index += 4, pixel += 1) {
        const luma =
          (imageData[index] * 54) +
          (imageData[index + 1] * 183) +
          (imageData[index + 2] * 19);

        const value = luma >> 8;

        if (this.hasPrev) {
          diffSum += Math.abs(value - this.prev[pixel]);
          pixelCount += 1;
        }

        this.prev[pixel] = value;
      }

      this.hasPrev = true;

      const normalized = pixelCount > 0 ? diffSum / (pixelCount * 255) : 0;
      this.motion = Math.min(1, normalized * 4.5);
      this.smoothed += (this.motion - this.smoothed) * 0.16;

      return this.smoothed;
    }
  }

  class GLRenderer {
    constructor(canvas, video) {
      this.canvas = canvas;
      this.video = video;
      this.name = "WebGL";
      this.gl =
        canvas.getContext("webgl", {
          alpha: false,
          antialias: false,
          depth: false,
          powerPreference: "high-performance",
          premultipliedAlpha: false,
          preserveDrawingBuffer: true,
          stencil: false,
        }) ||
        canvas.getContext("experimental-webgl", {
          alpha: false,
          antialias: false,
          depth: false,
          premultipliedAlpha: false,
          preserveDrawingBuffer: true,
          stencil: false,
        });

      if (!this.gl) {
        throw new Error("WebGL unavailable on this device.");
      }

      this.program = null;
      this.positionBuffer = null;
      this.textures = [];
      this.textureIndex = 0;
      this.frameCount = 0;
      this.uniforms = {};
      this.renderWidth = 0;
      this.renderHeight = 0;
      this.videoWidth = 0;
      this.videoHeight = 0;
      this.shouldUploadGhosts = true;
      this.lastGhostCaptureAt = -Infinity;

      this.init();
    }

    init() {
      const gl = this.gl;
      const vertexSource = `
        attribute vec2 a_position;
        varying vec2 v_uv;

        void main() {
          v_uv = a_position * 0.5 + 0.5;
          gl_Position = vec4(a_position, 0.0, 1.0);
        }
      `;

      const fragmentSource = `
        precision mediump float;

        varying vec2 v_uv;

        uniform sampler2D u_current;
        uniform sampler2D u_ghost0;
        uniform sampler2D u_ghost1;
        uniform sampler2D u_ghost2;
        uniform sampler2D u_ghost3;
        uniform vec2 u_resolution;
        uniform vec2 u_cover_scale;
        uniform vec2 u_sway;
        uniform float u_pinch;
        uniform float u_time;
        uniform float u_intensity;
        uniform float u_motion;

        float edgeMask(vec2 centeredUv) {
          return smoothstep(0.12, 0.86, length(centeredUv));
        }

        vec2 coverUv(vec2 uv) {
          vec2 centered = uv - 0.5;
          centered *= u_cover_scale;
          return centered + 0.5;
        }

        vec2 lensWarp(vec2 uv, float distortion, float wobble) {
          vec2 p = uv * 2.0 - 1.0;
          float r2 = dot(p, p);
          p *= 1.0 + distortion * r2;
          p.x += sin((uv.y * 5.2) + (u_time * 0.8)) * wobble;
          p.y += cos((uv.x * 4.1) + (u_time * 0.7)) * wobble * 0.75;
          return p * 0.5 + 0.5;
        }

        vec2 pinchWarp(vec2 uv, float pinch) {
          vec2 p = uv - 0.5;
          float r2 = dot(p, p);
          float strength = pinch * 0.38;
          p *= 1.0 - strength * (0.96 - r2);
          return p + 0.5;
        }

        vec2 transformGhostUv(vec2 uv, vec2 offset, float scale, float rotation) {
          vec2 p = uv - 0.5;
          float s = sin(rotation);
          float c = cos(rotation);
          mat2 rot = mat2(c, -s, s, c);
          p = rot * p;
          p *= scale;
          p += offset;
          return p + 0.5;
        }

        vec3 sampleTexture(sampler2D tex, vec2 uv) {
          return texture2D(tex, clamp(uv, 0.001, 0.999)).rgb;
        }

        vec3 sampleAberrated(sampler2D tex, vec2 uv, float edge, float amount) {
          vec2 centerDir = normalize(uv - 0.5 + 0.0001);
          vec2 shift = centerDir * amount * edge;

          float r = texture2D(tex, clamp(uv + shift, 0.001, 0.999)).r;
          float g = texture2D(tex, clamp(uv, 0.001, 0.999)).g;
          float b = texture2D(tex, clamp(uv - shift, 0.001, 0.999)).b;

          return vec3(r, g, b);
        }

        vec3 sampleBlurredCurrent(vec2 uv, float edge, float blurAmount) {
          vec2 dir = normalize(vec2(
            sin(u_time * 0.92) * 0.7 + u_motion * 1.1,
            cos(u_time * 0.63) * 0.4 + sin(u_time * 0.41) * 0.6
          ) + 0.0001);

          vec2 blurStep = dir * (1.0 / u_resolution) * (2.0 + 10.0 * blurAmount + 3.0 * edge);

          vec3 c0 = sampleAberrated(u_current, uv, edge, 0.0019 + edge * 0.0046);
          vec3 c1 = sampleAberrated(u_current, uv + blurStep, edge, 0.0019 + edge * 0.0046);
          vec3 c2 = sampleAberrated(u_current, uv - blurStep, edge, 0.0019 + edge * 0.0046);
          vec3 c3 = sampleAberrated(u_current, uv + blurStep * 1.7, edge, 0.0019 + edge * 0.0046);

          return (c0 * 0.46) + (c1 * 0.22) + (c2 * 0.22) + (c3 * 0.10);
        }

        void main() {
          vec2 sway = u_sway * (0.6 + edgeMask(v_uv * 2.0 - 1.0) * 0.45);
          vec2 coveredUv = coverUv(v_uv + sway);
          vec2 centered = coveredUv * 2.0 - 1.0;
          float edge = edgeMask(centered);
          float ghostStrength = 0.05 + u_intensity * 0.95;

          float distortion = 0.003 + u_intensity * 0.086;
          float wobble = 0.0004 + u_intensity * 0.0139;
          float focusPulse = 0.35 + 0.35 * sin(u_time * 1.6);
          float focusBlur = (0.002 + u_intensity * 0.278) * focusPulse;
          float motionBlur = min(0.54, u_motion * (0.02 + u_intensity * 0.62));
          float haze = 0.002 + u_intensity * 0.079;

          vec2 pinchedUv = pinchWarp(coveredUv, u_pinch);
          vec2 uv = lensWarp(pinchedUv, distortion, wobble);
          vec2 drift = vec2(
            sin(u_time * 0.82) * (0.0002 + u_intensity * 0.0058),
            cos(u_time * 0.67) * (0.0001 + u_intensity * 0.0041)
          );
          vec2 echoDrift = drift + (u_sway * 0.75);
          float ghostRot0 = 0.014 + sin(u_time * 0.73) * 0.005;
          float ghostRot1 = 0.022 + cos(u_time * 0.58) * 0.007;
          float ghostRot2 = 0.031 + sin(u_time * 0.49) * 0.0085;
          float ghostRot3 = 0.043 + cos(u_time * 0.41) * 0.010;

          vec3 current = sampleBlurredCurrent(uv, edge, focusBlur + motionBlur);
          vec3 ghost0 = sampleTexture(
            u_ghost0,
            transformGhostUv(uv, -echoDrift * 0.85, 1.012, ghostRot0)
          );
          vec3 ghost1 = sampleTexture(
            u_ghost1,
            transformGhostUv(uv, -echoDrift * 1.45, 1.022, ghostRot1)
          );
          vec3 ghost2 = sampleTexture(
            u_ghost2,
            transformGhostUv(uv, -echoDrift * 2.15, 1.034, ghostRot2)
          );
          vec3 ghost3 = sampleTexture(
            u_ghost3,
            transformGhostUv(uv, -echoDrift * 2.95, 1.048, ghostRot3)
          );

          vec3 color = current;
          color += ghost0 * (0.10 * (0.8 + u_intensity * 0.12) * ghostStrength);
          color += ghost1 * (0.16 * (0.9 + u_intensity * 0.12) * ghostStrength);
          color += ghost2 * (0.24 * (1.02 + u_intensity * 0.14) * ghostStrength);
          color += ghost3 * (0.52 * (1.08 + u_intensity * 0.16) * ghostStrength);

          vec3 bloom = max(current - vec3(0.48 - haze), 0.0) * (0.04 + u_intensity * 0.48);
          color += bloom;

          float luma = dot(color, vec3(0.299, 0.587, 0.114));
          color = mix(color, vec3(luma), 0.005 + (u_intensity * 0.085));
          color = mix(color, color + vec3(haze, haze * 0.82, haze * 0.68), 0.08 + (u_intensity * 0.48));
          color *= mix(0.92, 1.0, smoothstep(1.12, 0.22, length(centered)));
          color *= 1.0 + (edge * 0.02);

          gl_FragColor = vec4(min(color, 1.0), 1.0);
        }
      `;

      const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
      const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
      this.program = this.createProgram(vertexShader, fragmentShader);

      this.positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
          -1, -1,
           1, -1,
          -1,  1,
          -1,  1,
           1, -1,
           1,  1,
        ]),
        gl.STATIC_DRAW,
      );

      gl.useProgram(this.program);

      const aPosition = gl.getAttribLocation(this.program, "a_position");
      gl.enableVertexAttribArray(aPosition);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

      this.uniforms.resolution = gl.getUniformLocation(this.program, "u_resolution");
      this.uniforms.coverScale = gl.getUniformLocation(this.program, "u_cover_scale");
      this.uniforms.sway = gl.getUniformLocation(this.program, "u_sway");
      this.uniforms.pinch = gl.getUniformLocation(this.program, "u_pinch");
      this.uniforms.time = gl.getUniformLocation(this.program, "u_time");
      this.uniforms.intensity = gl.getUniformLocation(this.program, "u_intensity");
      this.uniforms.motion = gl.getUniformLocation(this.program, "u_motion");

      const samplers = ["u_current", "u_ghost0", "u_ghost1", "u_ghost2", "u_ghost3"];

      for (let index = 0; index < samplers.length; index += 1) {
        const texture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + index);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          2,
          2,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          new Uint8Array([
            0, 0, 0, 255, 0, 0, 0, 255,
            0, 0, 0, 255, 0, 0, 0, 255,
          ]),
        );
        gl.uniform1i(gl.getUniformLocation(this.program, samplers[index]), index);
        this.textures.push(texture);
      }
    }

    createShader(type, source) {
      const gl = this.gl;
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);

      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) || "Shader compile failed.";
        gl.deleteShader(shader);
        throw new Error(message);
      }

      return shader;
    }

    createProgram(vertexShader, fragmentShader) {
      const gl = this.gl;
      const program = gl.createProgram();
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program) || "Program link failed.";
        gl.deleteProgram(program);
        throw new Error(message);
      }

      return program;
    }

    setSize(width, height, videoWidth, videoHeight) {
      this.renderWidth = Math.max(2, Math.floor(width));
      this.renderHeight = Math.max(2, Math.floor(height));
      this.videoWidth = Math.max(2, Math.floor(videoWidth));
      this.videoHeight = Math.max(2, Math.floor(videoHeight));

      this.canvas.width = this.renderWidth;
      this.canvas.height = this.renderHeight;
      this.gl.viewport(0, 0, this.renderWidth, this.renderHeight);
    }

    updateVideoTextures(nowMs) {
      const gl = this.gl;

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.video);

      if (
        this.shouldUploadGhosts ||
        nowMs - this.lastGhostCaptureAt >= GHOST_CAPTURE_INTERVAL_MS
      ) {
        this.textureIndex = (this.textureIndex % MAX_GHOST_FRAMES) + 1;
        gl.activeTexture(gl.TEXTURE0 + this.textureIndex);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[this.textureIndex]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.video);
        this.shouldUploadGhosts = false;
        this.lastGhostCaptureAt = nowMs;
      }
    }

    render(timeSeconds, intensity, motionAmount, nowMs) {
      if (!this.renderWidth || !this.renderHeight) {
        return;
      }

      this.frameCount += 1;
      this.updateVideoTextures(nowMs);

      const renderAspect = this.renderWidth / this.renderHeight;
      const videoAspect = this.videoWidth / this.videoHeight;
      let coverScaleX = 1;
      let coverScaleY = 1;
      const sway = getSway(timeSeconds, intensity, motionAmount);
      const pinchAmount = getPinchAmount(timeSeconds);
      const swayUvX = sway.x / this.renderWidth;
      const swayUvY = sway.y / this.renderHeight;

      if (videoAspect > renderAspect) {
        coverScaleX = renderAspect / videoAspect;
      } else {
        coverScaleY = videoAspect / renderAspect;
      }

      const ghostIds = [];

      for (let index = 0; index < MAX_GHOST_FRAMES; index += 1) {
        const ghostIndex =
          ((this.textureIndex - index - 1 + MAX_GHOST_FRAMES) % MAX_GHOST_FRAMES) + 1;
        ghostIds.push(ghostIndex);
      }

      this.gl.useProgram(this.program);
      this.gl.uniform2f(this.uniforms.resolution, this.renderWidth, this.renderHeight);
      this.gl.uniform2f(this.uniforms.coverScale, coverScaleX, coverScaleY);
      this.gl.uniform2f(this.uniforms.sway, swayUvX, swayUvY);
      this.gl.uniform1f(this.uniforms.pinch, pinchAmount);
      this.gl.uniform1f(this.uniforms.time, timeSeconds);
      this.gl.uniform1f(this.uniforms.intensity, intensity);
      this.gl.uniform1f(this.uniforms.motion, motionAmount);

      for (let index = 0; index < ghostIds.length; index += 1) {
        this.gl.activeTexture(this.gl.TEXTURE0 + index + 1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.textures[ghostIds[index]]);
      }

      this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }
  }

  class CanvasRenderer {
    constructor(canvas, video) {
      this.canvas = canvas;
      this.video = video;
      this.name = "Canvas2D";
      this.ctx = create2DContext(canvas, { alpha: false, desynchronized: true });

      if (!this.ctx) {
        throw new Error("Canvas2D unavailable on this device.");
      }

      this.ctx.imageSmoothingEnabled = true;
      this.currentFrame = document.createElement("canvas");
      this.currentCtx = create2DContext(this.currentFrame, { alpha: false });
      this.lagFrame = document.createElement("canvas");
      this.lagCtx = create2DContext(this.lagFrame, { alpha: false });
      this.lagScratch = document.createElement("canvas");
      this.lagScratchCtx = create2DContext(this.lagScratch, { alpha: false });
      this.pinchFrame = document.createElement("canvas");
      this.pinchCtx = create2DContext(this.pinchFrame, { alpha: false });
      this.pinchScratch = document.createElement("canvas");
      this.pinchScratchCtx = create2DContext(this.pinchScratch, { alpha: false });
      this.ghostFrames = Array.from({ length: MAX_GHOST_FRAMES }, () => {
        const frame = document.createElement("canvas");
        return {
          canvas: frame,
          ctx: create2DContext(frame, { alpha: false }),
        };
      });

      if (
        !this.currentCtx ||
        !this.lagCtx ||
        !this.lagScratchCtx ||
        !this.pinchCtx ||
        !this.pinchScratchCtx ||
        this.ghostFrames.some((ghost) => !ghost.ctx)
      ) {
        throw new Error("Canvas buffers unavailable on this device.");
      }

      this.ghostIndex = 0;
      this.frameCount = 0;
      this.lastGhostCaptureAt = -Infinity;
      this.renderWidth = 0;
      this.renderHeight = 0;
      this.videoWidth = 0;
      this.videoHeight = 0;
      this.vignette = null;
    }

    setSize(width, height, videoWidth, videoHeight) {
      this.renderWidth = Math.max(2, Math.floor(width));
      this.renderHeight = Math.max(2, Math.floor(height));
      this.videoWidth = Math.max(2, Math.floor(videoWidth));
      this.videoHeight = Math.max(2, Math.floor(videoHeight));

      this.canvas.width = this.renderWidth;
      this.canvas.height = this.renderHeight;
      this.currentFrame.width = this.renderWidth;
      this.currentFrame.height = this.renderHeight;
      this.lagFrame.width = this.renderWidth;
      this.lagFrame.height = this.renderHeight;
      this.lagScratch.width = this.renderWidth;
      this.lagScratch.height = this.renderHeight;
      this.pinchFrame.width = this.renderWidth;
      this.pinchFrame.height = this.renderHeight;
      this.pinchScratch.width = this.renderWidth;
      this.pinchScratch.height = this.renderHeight;

      for (const ghost of this.ghostFrames) {
        ghost.canvas.width = this.renderWidth;
        ghost.canvas.height = this.renderHeight;
      }

      this.vignette = this.createVignette();
    }

    createVignette() {
      const gradient = this.ctx.createRadialGradient(
        this.renderWidth * 0.5,
        this.renderHeight * 0.48,
        this.renderWidth * 0.08,
        this.renderWidth * 0.5,
        this.renderHeight * 0.5,
        Math.max(this.renderWidth, this.renderHeight) * 0.72,
      );

      gradient.addColorStop(0, "rgba(0, 0, 0, 0)");
      gradient.addColorStop(0.62, "rgba(0, 0, 0, 0.02)");
      gradient.addColorStop(1, "rgba(6, 8, 12, 0.28)");
      return gradient;
    }

    updateCurrentFrame(nowMs) {
      const crop = getCoverCrop(
        this.videoWidth,
        this.videoHeight,
        this.renderWidth,
        this.renderHeight,
      );

      this.currentCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.currentCtx.globalCompositeOperation = "source-over";
      this.currentCtx.globalAlpha = 1;
      this.currentCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);
      this.currentCtx.drawImage(
        this.video,
        crop.sx,
        crop.sy,
        crop.sWidth,
        crop.sHeight,
        0,
        0,
        this.renderWidth,
        this.renderHeight,
      );

      if (this.frameCount === 1) {
        this.lagCtx.setTransform(1, 0, 0, 1, 0, 0);
        this.lagCtx.globalCompositeOperation = "source-over";
        this.lagCtx.globalAlpha = 1;
        this.lagCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);
        this.lagCtx.drawImage(this.currentFrame, 0, 0);
      } else {
        this.lagScratchCtx.setTransform(1, 0, 0, 1, 0, 0);
        this.lagScratchCtx.globalCompositeOperation = "source-over";
        this.lagScratchCtx.globalAlpha = 1;
        this.lagScratchCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);

        this.lagScratchCtx.globalAlpha = 1 - TEMPORAL_LAG_BLEND;
        this.lagScratchCtx.drawImage(this.lagFrame, 0, 0);
        this.lagScratchCtx.globalAlpha = TEMPORAL_LAG_BLEND;
        this.lagScratchCtx.drawImage(this.currentFrame, 0, 0);

        this.lagCtx.setTransform(1, 0, 0, 1, 0, 0);
        this.lagCtx.globalCompositeOperation = "source-over";
        this.lagCtx.globalAlpha = 1;
        this.lagCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);
        this.lagCtx.drawImage(this.lagScratch, 0, 0);
      }

      if (
        this.frameCount === 1 ||
        nowMs - this.lastGhostCaptureAt >= GHOST_CAPTURE_INTERVAL_MS
      ) {
        const ghost = this.ghostFrames[this.ghostIndex];
        ghost.ctx.setTransform(1, 0, 0, 1, 0, 0);
        ghost.ctx.globalCompositeOperation = "source-over";
        ghost.ctx.globalAlpha = 1;
        ghost.ctx.clearRect(0, 0, this.renderWidth, this.renderHeight);
        ghost.ctx.drawImage(this.lagFrame, 0, 0);
        this.ghostIndex = (this.ghostIndex + 1) % MAX_GHOST_FRAMES;
        this.lastGhostCaptureAt = nowMs;
      }
    }

    drawLayer(source, options) {
      const {
        alpha,
        offsetX,
        offsetY,
        scaleX,
        scaleY,
        rotation = 0,
        blendMode = "source-over",
        blur = 0,
        brightness = 1,
      } = options;

      const width = this.renderWidth * scaleX;
      const height = this.renderHeight * scaleY;
      const centerX = (this.renderWidth * 0.5) + offsetX;
      const centerY = (this.renderHeight * 0.5) + offsetY;

      this.ctx.save();
      this.ctx.globalCompositeOperation = blendMode;
      this.ctx.globalAlpha = Math.max(0, alpha);
      this.ctx.filter = `blur(${blur.toFixed(2)}px) brightness(${brightness.toFixed(3)})`;
      this.ctx.translate(centerX, centerY);
      this.ctx.rotate(rotation);
      this.ctx.drawImage(source, -width * 0.5, -height * 0.5, width, height);
      this.ctx.restore();
    }

    applyPinchWarp(source, pinchAmount) {
      const horizontalSlices = 26;
      const verticalSlices = 18;
      const pinchStrength = pinchAmount * 0.38;
      const sliceHeight = this.renderHeight / horizontalSlices;
      const sliceWidth = this.renderWidth / verticalSlices;

      this.pinchScratchCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.pinchScratchCtx.globalCompositeOperation = "source-over";
      this.pinchScratchCtx.globalAlpha = 1;
      this.pinchScratchCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);

      for (let index = 0; index < horizontalSlices; index += 1) {
        const sy = index * sliceHeight;
        const centerY = (sy + sliceHeight * 0.5) / this.renderHeight;
        const influence = 1 - Math.min(1, Math.abs((centerY - 0.5) * 2));
        const scaleX = 1 - (pinchStrength * Math.pow(influence, 1.35));
        const width = this.renderWidth * scaleX;
        const dx = (this.renderWidth - width) * 0.5;

        this.pinchScratchCtx.drawImage(
          source,
          0,
          sy,
          this.renderWidth,
          sliceHeight,
          dx,
          sy,
          width,
          sliceHeight + 1,
        );
      }

      this.pinchCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.pinchCtx.globalCompositeOperation = "source-over";
      this.pinchCtx.globalAlpha = 1;
      this.pinchCtx.clearRect(0, 0, this.renderWidth, this.renderHeight);

      for (let index = 0; index < verticalSlices; index += 1) {
        const sx = index * sliceWidth;
        const centerX = (sx + sliceWidth * 0.5) / this.renderWidth;
        const influence = 1 - Math.min(1, Math.abs((centerX - 0.5) * 2));
        const scaleY = 1 - (pinchStrength * Math.pow(influence, 1.35));
        const height = this.renderHeight * scaleY;
        const dy = (this.renderHeight - height) * 0.5;

        this.pinchCtx.drawImage(
          this.pinchScratch,
          sx,
          0,
          sliceWidth,
          this.renderHeight,
          sx,
          dy,
          sliceWidth + 1,
          height,
        );
      }

      return this.pinchFrame;
    }

    drawBloom(source, intensity, blurRadius) {
      this.ctx.save();
      this.ctx.globalCompositeOperation = "screen";
      this.ctx.globalAlpha = 0.02 + intensity * 0.3;
      this.ctx.filter = `blur(${blurRadius.toFixed(2)}px) brightness(${(1.16 + intensity * 0.18).toFixed(3)})`;
      this.ctx.drawImage(source, 0, 0, this.renderWidth, this.renderHeight);
      this.ctx.restore();
    }

    drawEdgeFringe(source, intensity) {
      const amount = 0.2 + intensity * 6.4;
      const centerX = this.renderWidth * 0.5;
      const centerY = this.renderHeight * 0.5;
      const radius = Math.max(this.renderWidth, this.renderHeight) * 0.72;
      const warmGlow = this.ctx.createRadialGradient(centerX, centerY, radius * 0.34, centerX, centerY, radius);
      const coolGlow = this.ctx.createRadialGradient(centerX, centerY, radius * 0.28, centerX, centerY, radius * 1.02);

      this.ctx.save();
      this.ctx.globalCompositeOperation = "screen";
      this.ctx.globalAlpha = 0.01 + intensity * 0.19;
      this.ctx.drawImage(source, -amount, 0, this.renderWidth, this.renderHeight);
      warmGlow.addColorStop(0, "rgba(255, 92, 72, 0)");
      warmGlow.addColorStop(0.58, "rgba(255, 92, 72, 0)");
      warmGlow.addColorStop(1, "rgba(255, 92, 72, 0.18)");
      this.ctx.fillStyle = warmGlow;
      this.ctx.fillRect(0, 0, this.renderWidth, this.renderHeight);

      this.ctx.globalAlpha = 0.01 + intensity * 0.145;
      this.ctx.drawImage(source, amount, amount * 0.4, this.renderWidth, this.renderHeight);
      coolGlow.addColorStop(0, "rgba(92, 126, 255, 0)");
      coolGlow.addColorStop(0.54, "rgba(92, 126, 255, 0)");
      coolGlow.addColorStop(1, "rgba(92, 126, 255, 0.16)");
      this.ctx.fillStyle = coolGlow;
      this.ctx.fillRect(0, 0, this.renderWidth, this.renderHeight);
      this.ctx.restore();
    }

    drawDistortionOverlay(source, timeSeconds, intensity, motionAmount) {
      const strips = 14;
      const sliceHeight = this.renderHeight / strips;
      const amplitude = 0.12 + intensity * 6.28 + motionAmount * (0.2 + intensity * 1.7);

      this.ctx.save();
      this.ctx.globalCompositeOperation = "screen";
      this.ctx.globalAlpha = 0.015 + intensity * 0.255;

      for (let index = 0; index < strips; index += 1) {
        const sy = index * sliceHeight;
        const wave =
          Math.sin((timeSeconds * 0.9) + (index * 0.55)) * amplitude +
          Math.cos((timeSeconds * 0.38) + (index * 0.21)) * amplitude * 0.5;
        const edgeBias = Math.abs(((index / (strips - 1)) * 2) - 1);
        const scaleX = 1 + (edgeBias * intensity * 0.018);
        const width = this.renderWidth * scaleX;
        const dx = ((this.renderWidth - width) * 0.5) + wave;

        this.ctx.drawImage(
          source,
          0,
          sy,
          this.renderWidth,
          sliceHeight,
          dx,
          sy,
          width,
          sliceHeight + 1,
        );
      }

      this.ctx.restore();
    }

    render(timeSeconds, intensity, motionAmount, nowMs) {
      if (!this.renderWidth || !this.renderHeight) {
        return;
      }

      this.frameCount += 1;
      this.updateCurrentFrame(nowMs);

      const sway = getSway(timeSeconds, intensity, motionAmount);
      const jitter = getWobbleJitter(timeSeconds, intensity);
      const pinchAmount = getPinchAmount(timeSeconds, intensity);
      const ghostStrength = 0.04 + intensity * 0.96;
      const blurStrength = 0.03 + intensity * 0.97;
      const primaryFrame = this.applyPinchWarp(this.lagFrame, pinchAmount);
      const swayX = sway.x * (3.1 + intensity * 1.6);
      const swayY = sway.y * (2.6 + intensity * 1.35);
      const driftX = swayX + jitter.x + (Math.sin(timeSeconds * 0.82) * (0.2 + intensity * 19.1));
      const driftY = swayY + jitter.y + (Math.cos(timeSeconds * 0.67) * (0.15 + intensity * 12.05));
      const focusPulse = 0.35 + 0.35 * Math.sin(timeSeconds * 1.6);
      const blurAmount =
        (0.08 + intensity * 5.02) *
        (0.35 + (focusPulse * (0.25 + intensity * 0.75)) + motionAmount * (0.08 + intensity * 1.47));
      const dreamyBlur =
        lerp(0.18, 1.4, intensity) +
        (focusPulse * lerp(0.25, 1.9, intensity)) +
        (intensity * 2.05);
      const dirX = Math.sin(timeSeconds * 0.92) * 0.75 + motionAmount * 1.15;
      const dirY = Math.cos(timeSeconds * 0.63) * 0.45 + Math.sin(timeSeconds * 0.41) * 0.6;
      const magnitude = Math.hypot(dirX, dirY) || 1;
      const normX = dirX / magnitude;
      const normY = dirY / magnitude;
      const swayScale = 1.001 + (intensity * 0.051);
      const baseRotation = jitter.rotation;

      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.globalCompositeOperation = "source-over";
      this.ctx.globalAlpha = 1;
      this.ctx.clearRect(0, 0, this.renderWidth, this.renderHeight);
      this.ctx.fillStyle = "#000";
      this.ctx.fillRect(0, 0, this.renderWidth, this.renderHeight);

      for (let index = 0; index < MAX_GHOST_FRAMES; index += 1) {
        const ghost = this.ghostFrames[
          (this.ghostIndex - index - 1 + MAX_GHOST_FRAMES) % MAX_GHOST_FRAMES
        ];
        const weight = (GHOST_WEIGHTS[index] || 0.08) * ghostStrength;
        const offsetFactor = 1.05 + (index * 0.9);
        const rotation = baseRotation + ((index + 1) * 0.018 * (0.85 + intensity));
        const layerBlur =
          (dreamyBlur * ghostStrength) +
          (index * (0.08 + intensity * 0.97)) +
          (motionAmount * (0.2 + intensity * 1.9));

        this.drawLayer(ghost.canvas, {
          alpha: weight,
          offsetX: -driftX * offsetFactor,
          offsetY: -driftY * offsetFactor,
          scaleX: swayScale + 0.022 + (index * 0.02),
          scaleY: swayScale + 0.015 + (index * 0.018),
          rotation,
          blendMode: "lighter",
          blur: layerBlur,
          brightness: 1.08 + (weight * 0.22),
        });
      }

      this.drawLayer(primaryFrame, {
        alpha: 0.88,
        offsetX: swayX * 0.72,
        offsetY: swayY * 0.72,
        scaleX: swayScale,
        scaleY: swayScale,
        rotation: baseRotation * 0.95,
        blendMode: "source-over",
        blur: dreamyBlur * (0.08 + intensity * 0.5),
        brightness: 1.02,
      });

      this.drawLayer(this.currentFrame, {
        alpha: 0.03 + intensity * 0.15,
        offsetX: swayX * 0.22,
        offsetY: swayY * 0.22,
        scaleX: 1.002,
        scaleY: 1.002,
        rotation: baseRotation * 0.28,
        blendMode: "screen",
        blur: dreamyBlur * (0.02 + intensity * 0.13),
        brightness: 1.04,
      });

      const blurOffsets = [-2.4, -1.35, -0.55, 0, 1.15, 2.2];
      const blurAlphas = [0.12, 0.18, 0.22, 0.4, 0.18, 0.1];

      for (let index = 0; index < blurOffsets.length; index += 1) {
        const shift = blurOffsets[index] * blurAmount;
        this.drawLayer(primaryFrame, {
          alpha: blurAlphas[index] * blurStrength,
          offsetX: swayX + (normX * shift) + (Math.sin(timeSeconds * 0.8) * intensity * 2),
          offsetY: swayY + (normY * shift) + (Math.cos(timeSeconds * 0.66) * intensity * 1.4),
          scaleX: swayScale,
          scaleY: swayScale,
          rotation: baseRotation * (0.95 + index * 0.11),
          blendMode: "screen",
          blur: (dreamyBlur * blurStrength) + Math.abs(shift) * (0.04 + intensity * 0.28),
          brightness: 1.05 + (blurAlphas[index] * 0.12),
        });
      }

      this.drawBloom(primaryFrame, intensity, 14 + (focusPulse * 12));
      this.drawEdgeFringe(primaryFrame, intensity);
      this.drawDistortionOverlay(primaryFrame, timeSeconds, intensity, motionAmount);

      this.ctx.globalCompositeOperation = "source-over";
      this.ctx.globalAlpha = 1;
      this.ctx.fillStyle = "rgba(26, 20, 16, 0.06)";
      this.ctx.fillRect(0, 0, this.renderWidth, this.renderHeight);
      this.ctx.fillStyle = this.vignette;
      this.ctx.fillRect(0, 0, this.renderWidth, this.renderHeight);
    }
  }

  class DrunkVisionApp {
    constructor() {
      this.video = document.getElementById("camera");
      this.canvas = document.getElementById("drunk-canvas");
      this.overlay = document.querySelector(".overlay");
      this.controlsToggle = document.getElementById("controls-toggle");
      this.startButton = document.getElementById("start-button");
      this.vrToggle = document.getElementById("vr-toggle");
      this.flipToggle = document.getElementById("flip-toggle");
      this.fullscreenToggle = document.getElementById("fullscreen-toggle");
      this.intensitySlider = document.getElementById("intensity-slider");
      this.statusBadge = document.getElementById("status-badge");
      this.renderCanvas = document.createElement("canvas");
      this.displayCtx = create2DContext(this.canvas, { alpha: false, desynchronized: true });

      this.motionEstimator = new MotionEstimator();
      this.renderer = null;
      this.stream = null;
      this.rafId = 0;
      this.running = false;
      this.vrMode = true;
      this.flipMode = true;
      this.hasRenderedFrame = false;
      this.controlsVisible = true;
      this.controlsHideTimer = 0;
      this.startTime = performance.now();
      this.intensity = this.readIntensity();

      this.boundRender = this.render.bind(this);
      this.boundResize = this.handleResize.bind(this);
      this.boundVisibility = this.handleVisibilityChange.bind(this);
      this.boundFullscreenChange = this.handleFullscreenChange.bind(this);

      this.setup();
    }

    setup() {
      this.startButton.addEventListener("click", () => {
        this.startCamera();
      });

      this.vrToggle.addEventListener("click", () => {
        this.toggleVrMode();
      });

      this.flipToggle.addEventListener("click", () => {
        this.toggleFlipMode();
      });

      this.fullscreenToggle.addEventListener("click", () => {
        this.toggleFullscreen();
      });

      this.controlsToggle.addEventListener("click", () => {
        this.showControls();
      });

      this.intensitySlider.addEventListener("input", () => {
        this.intensity = this.readIntensity();
        this.updateStatus(
          this.running
            ? `Intensity ${(this.intensity * 100).toFixed(0)}%`
            : "Adjust intensity, then start camera",
        );

        if (this.running) {
          this.scheduleControlsHide();
        }
      });

      window.addEventListener("resize", this.boundResize, { passive: true });
      window.addEventListener("orientationchange", this.boundResize, { passive: true });
      document.addEventListener("visibilitychange", this.boundVisibility);
      document.addEventListener("fullscreenchange", this.boundFullscreenChange);
      document.addEventListener("webkitfullscreenchange", this.boundFullscreenChange);

      this.updateVrToggle();
      this.updateFlipToggle();
      this.updateFullscreenToggle();
      this.drawIdleBackdrop();
    }

    readIntensity() {
      return Number(this.intensitySlider.value) / 100;
    }

    updateStatus(message) {
      this.statusBadge.textContent = message;
    }

    updateVrToggle() {
      if (!this.vrToggle) {
        return;
      }

      this.vrToggle.textContent = this.vrMode ? "VR Split: On" : "VR Split: Off";
      this.vrToggle.setAttribute("aria-pressed", this.vrMode ? "true" : "false");
    }

    updateFlipToggle() {
      if (!this.flipToggle) {
        return;
      }

      this.flipToggle.textContent = this.flipMode ? "Rotate 180: On" : "Rotate 180: Off";
      this.flipToggle.setAttribute("aria-pressed", this.flipMode ? "true" : "false");
    }

    isFullscreenActive() {
      return Boolean(
        document.fullscreenElement ||
        document.webkitFullscreenElement,
      );
    }

    updateFullscreenToggle() {
      if (!this.fullscreenToggle) {
        return;
      }

      const active = this.isFullscreenActive();
      this.fullscreenToggle.textContent = active ? "Exit Full Screen" : "Full Screen";
      this.fullscreenToggle.setAttribute("aria-pressed", active ? "true" : "false");
    }

    getCameraActiveMessage() {
      if (!this.renderer) {
        return this.vrMode ? "Rear camera active - VR split" : "Rear camera active";
      }

      if (this.renderer.name === "Canvas2D") {
        return this.vrMode
          ? "Rear camera active - VR split - Canvas fallback"
          : "Rear camera active - Canvas fallback";
      }

      return this.vrMode ? "Rear camera active - VR split" : "Rear camera active";
    }

    toggleVrMode() {
      this.vrMode = !this.vrMode;
      this.updateVrToggle();
      this.handleResize();

      if (!this.running) {
        this.drawIdleBackdrop();
      } else {
        this.updateStatus(this.getCameraActiveMessage());
        this.scheduleControlsHide();
      }
    }

    toggleFlipMode() {
      this.flipMode = !this.flipMode;
      this.updateFlipToggle();

      if (!this.running) {
        this.drawIdleBackdrop();
      } else {
        this.updateStatus(this.getCameraActiveMessage());
        this.scheduleControlsHide();
      }
    }

    async requestFullscreenMode() {
      if (this.isFullscreenActive()) {
        return true;
      }

      const root = document.documentElement;

      try {
        if (root.requestFullscreen) {
          await root.requestFullscreen({ navigationUI: "hide" });
        } else if (root.webkitRequestFullscreen) {
          root.webkitRequestFullscreen();
        } else {
          return false;
        }

        this.updateFullscreenToggle();
        return true;
      } catch (error) {
        console.warn("Unable to enter fullscreen.", error);
        this.updateFullscreenToggle();
        return false;
      }
    }

    async exitFullscreenMode() {
      try {
        if (document.exitFullscreen && document.fullscreenElement) {
          await document.exitFullscreen();
        } else if (document.webkitExitFullscreen && document.webkitFullscreenElement) {
          document.webkitExitFullscreen();
        }
      } catch (error) {
        console.warn("Unable to exit fullscreen.", error);
      } finally {
        this.updateFullscreenToggle();
      }
    }

    async toggleFullscreen() {
      if (this.isFullscreenActive()) {
        await this.exitFullscreenMode();
      } else {
        const entered = await this.requestFullscreenMode();

        if (!entered && !this.running) {
          this.updateStatus("Fullscreen unavailable on this browser");
        }
      }

      if (this.running) {
        this.scheduleControlsHide();
      }
    }

    showControls() {
      clearTimeout(this.controlsHideTimer);
      this.controlsVisible = true;
      this.overlay.classList.remove("controls-hidden");
      this.controlsToggle.setAttribute("aria-expanded", "true");

      if (this.running) {
        this.scheduleControlsHide(2200);
      }
    }

    hideControls() {
      if (!this.running) {
        return;
      }

      this.controlsVisible = false;
      this.overlay.classList.add("controls-hidden");
      this.controlsToggle.setAttribute("aria-expanded", "false");
    }

    scheduleControlsHide(delay = 1400) {
      clearTimeout(this.controlsHideTimer);
      this.controlsHideTimer = window.setTimeout(() => {
        this.hideControls();
      }, delay);
    }

    drawIdleBackdrop() {
      const width = Math.max(window.innerWidth, 320);
      const height = Math.max(window.innerHeight, 568);
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

      this.canvas.width = Math.floor(width * dpr);
      this.canvas.height = Math.floor(height * dpr);

      if (!this.displayCtx) {
        return;
      }

      const gradient = this.displayCtx.createRadialGradient(
        this.canvas.width * 0.5,
        this.canvas.height * 0.36,
        this.canvas.width * 0.08,
        this.canvas.width * 0.5,
        this.canvas.height * 0.5,
        this.canvas.height * 0.85,
      );

      gradient.addColorStop(0, "rgba(245, 192, 109, 0.18)");
      gradient.addColorStop(0.42, "rgba(29, 38, 54, 0.22)");
      gradient.addColorStop(1, "rgba(5, 7, 10, 0.96)");

      this.displayCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.displayCtx.globalCompositeOperation = "source-over";
      this.displayCtx.globalAlpha = 1;
      this.displayCtx.fillStyle = "#000";
      this.displayCtx.fillRect(0, 0, this.canvas.width, this.canvas.height);

      if (this.vrMode) {
        const eyeWidth = this.canvas.width * 0.5;
        this.drawDisplaySegment((ctx) => {
          ctx.fillStyle = gradient;
          ctx.fillRect(0, 0, eyeWidth, this.canvas.height);
          ctx.fillRect(eyeWidth, 0, eyeWidth, this.canvas.height);
        }, this.flipMode);
        this.displayCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
        this.displayCtx.fillRect(eyeWidth - 1, 0, 2, this.canvas.height);
        return;
      }

      this.drawDisplaySegment((ctx) => {
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      }, this.flipMode);
    }

    async startCamera() {
      if (this.running) {
        return;
      }

      this.startButton.disabled = true;
      this.startButton.textContent = "Starting…";
      this.updateStatus("Requesting camera permission");

      await this.requestFullscreenMode();

      const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1280;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 720;
      const targetAspectRatio = Math.max(0.5, Math.min(2, viewportWidth / Math.max(viewportHeight, 1)));

      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: { ideal: "environment" },
            aspectRatio: { ideal: targetAspectRatio },
            width: { ideal: Math.max(viewportWidth * 1.5, 960) },
            height: { ideal: Math.max(viewportHeight * 1.5, 720) },
          },
        });

        this.video.srcObject = this.stream;
        await this.video.play();
        await this.waitForVideoMetadata();
        await this.waitForVideoFrame();

        this.ensureRenderer();
        this.handleResize();
        this.running = true;
        this.hasRenderedFrame = false;
        this.startTime = performance.now();
        this.startButton.textContent = "Camera Live";
        this.updateStatus("Camera ready - waiting for first frame");
        this.render(performance.now());
      } catch (error) {
        console.error(error);
        this.startButton.disabled = false;
        this.startButton.textContent = "Start Camera";
        this.updateStatus(this.describeCameraError(error));
      }
    }

    ensureRenderer() {
      if (this.renderer) {
        return;
      }

      try {
        // Offscreen WebGL-to-2D compositing is unreliable on mobile Safari,
        // so prefer the Canvas renderer for the VR split display pipeline.
        this.renderer = new CanvasRenderer(this.renderCanvas, this.video);
      } catch (canvasError) {
        console.warn("Canvas2D renderer unavailable. Falling back to WebGL.", canvasError);
        this.renderer = new GLRenderer(this.renderCanvas, this.video);
      }
    }

    waitForVideoMetadata() {
      if (this.video.readyState >= HTMLMediaElement.HAVE_METADATA && this.video.videoWidth > 0) {
        return Promise.resolve();
      }

      return new Promise((resolve) => {
        const onLoaded = () => {
          this.video.removeEventListener("loadedmetadata", onLoaded);
          resolve();
        };

        this.video.addEventListener("loadedmetadata", onLoaded);
      });
    }

    waitForVideoFrame() {
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        return Promise.resolve();
      }

      return new Promise((resolve, reject) => {
        let settled = false;

        const cleanup = () => {
          window.clearTimeout(timeoutId);
          this.video.removeEventListener("loadeddata", onReady);
          this.video.removeEventListener("canplay", onReady);
          this.video.removeEventListener("playing", onReady);
        };

        const onReady = () => {
          if (settled || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            return;
          }

          settled = true;
          cleanup();
          resolve();
        };

        const timeoutId = window.setTimeout(() => {
          if (settled) {
            return;
          }

          settled = true;
          cleanup();
          reject(new Error("Camera stream started, but no video frames arrived."));
        }, 5000);

        this.video.addEventListener("loadeddata", onReady);
        this.video.addEventListener("canplay", onReady);
        this.video.addEventListener("playing", onReady);
      });
    }

    describeCameraError(error) {
      if (!error || !error.name) {
        return error && error.message ? error.message : "Camera failed to start";
      }

      switch (error.name) {
        case "NotAllowedError":
        case "SecurityError":
          return "Camera blocked. Use HTTPS and allow access.";
        case "NotFoundError":
        case "OverconstrainedError":
          return "Rear camera unavailable on this device.";
        default:
          return error.message || "Unable to access camera";
      }
    }

    handleResize() {
      const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

      this.canvas.width = Math.max(2, Math.floor(viewportWidth * dpr));
      this.canvas.height = Math.max(2, Math.floor(viewportHeight * dpr));

      if (!this.renderer || !this.video.videoWidth || !this.video.videoHeight) {
        if (!this.running) {
          this.drawIdleBackdrop();
        }
        return;
      }

      const eyeWidth = this.vrMode ? viewportWidth * 0.5 : viewportWidth;

      this.renderer.setSize(
        eyeWidth * dpr,
        viewportHeight * dpr,
        this.video.videoWidth,
        this.video.videoHeight,
      );

      if (!this.running) {
        this.drawIdleBackdrop();
      }
    }

    handleVisibilityChange() {
      if (!this.running) {
        return;
      }

      if (document.hidden) {
        cancelAnimationFrame(this.rafId);
        this.rafId = 0;
        clearTimeout(this.controlsHideTimer);
        this.updateStatus("Paused while app is hidden");
        return;
      }

      this.startTime = performance.now();
      this.hasRenderedFrame = false;
      this.updateStatus(this.getCameraActiveMessage());
      this.scheduleControlsHide(1200);
      this.render(performance.now());
    }

    handleFullscreenChange() {
      this.updateFullscreenToggle();

      if (!this.running) {
        return;
      }

      this.handleResize();
      this.scheduleControlsHide(1200);
    }

    compositeFrame() {
      if (!this.displayCtx || !this.renderer) {
        return;
      }

      const displayWidth = this.canvas.width;
      const displayHeight = this.canvas.height;

      this.displayCtx.setTransform(1, 0, 0, 1, 0, 0);
      this.displayCtx.globalCompositeOperation = "source-over";
      this.displayCtx.globalAlpha = 1;
      this.displayCtx.fillStyle = "#000";
      this.displayCtx.fillRect(0, 0, displayWidth, displayHeight);

      if (this.vrMode) {
        const eyeWidth = Math.floor(displayWidth * 0.5);

        this.drawDisplaySegment((ctx) => {
          ctx.drawImage(this.renderCanvas, 0, 0, eyeWidth, displayHeight);
          ctx.drawImage(this.renderCanvas, eyeWidth, 0, eyeWidth, displayHeight);
        }, this.flipMode);
        this.displayCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
        this.displayCtx.fillRect(eyeWidth - 1, 0, 2, displayHeight);
        return;
      }

      this.drawDisplaySegment((ctx) => {
        ctx.drawImage(this.renderCanvas, 0, 0, displayWidth, displayHeight);
      }, this.flipMode);
    }

    drawDisplaySegment(drawFn, shouldFlip = false) {
      if (!this.displayCtx) {
        return;
      }

      this.displayCtx.save();

      if (shouldFlip) {
        this.displayCtx.translate(this.canvas.width, this.canvas.height);
        this.displayCtx.rotate(Math.PI);
      }

      drawFn(this.displayCtx);
      this.displayCtx.restore();
    }

    render(now) {
      if (!this.running || !this.renderer || document.hidden) {
        return;
      }

      if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.showControls();
        this.updateStatus("Waiting for camera frames");
        this.rafId = requestAnimationFrame(this.boundRender);
        return;
      }

      if (!this.hasRenderedFrame) {
        this.hasRenderedFrame = true;
        this.updateStatus(this.getCameraActiveMessage());
        this.scheduleControlsHide(900);
      }

      const elapsed = (now - this.startTime) / 1000;
      const motionAmount = this.motionEstimator.sample(this.video);
      this.renderer.render(elapsed, getEffectLevel(this.intensity), motionAmount, now);
      this.compositeFrame();
      this.rafId = requestAnimationFrame(this.boundRender);
    }
  }

  if (!window.isSecureContext) {
    document.getElementById("status-badge").textContent = "Camera requires HTTPS or localhost";
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    document.getElementById("status-badge").textContent = "Camera API unavailable in this browser";
    return;
  }

  new DrunkVisionApp();
})();
