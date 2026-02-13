/**
 * animations.js — Mouse-Interactive Particle Background
 *
 * Creates a full-screen canvas with:
 * - Floating particles that follow the mouse cursor
 * - Lines connecting nearby particles
 * - Smooth glowing effect with blend modes
 * - Parallax depth effect
 * - 60fps rendering via requestAnimationFrame
 */

(function () {
    'use strict';

    // ---- Configuration ----
    const CONFIG = {
        particleCount: 80,         // Number of particles
        connectDistance: 140,       // Max distance to draw connections
        mouseInfluence: 200,       // Mouse attraction radius
        mouseForce: 0.03,          // Mouse attraction strength
        friction: 0.97,            // Velocity damping
        baseSpeed: 0.3,            // Base particle speed
        minSize: 1.5,              // Min particle radius
        maxSize: 3.5,              // Max particle radius
        colors: [
            'rgba(0, 229, 255, ',  // Cyan
            'rgba(179, 136, 255, ', // Purple
            'rgba(255, 64, 129, ',  // Pink
        ],
        lineOpacity: 0.12,
        glowBlur: 60,
    };

    // ---- Canvas Setup ----
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height;
    let mouseX = -1000, mouseY = -1000;
    let particles = [];

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    // ---- Particle Class ----
    class Particle {
        constructor() {
            this.reset();
        }

        reset() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.vy = (Math.random() - 0.5) * CONFIG.baseSpeed;
            this.size = CONFIG.minSize + Math.random() * (CONFIG.maxSize - CONFIG.minSize);
            this.depth = 0.3 + Math.random() * 0.7; // Parallax depth (0.3 to 1.0)
            this.colorIndex = Math.floor(Math.random() * CONFIG.colors.length);
            this.alpha = 0.3 + Math.random() * 0.5;
        }

        update() {
            // Mouse attraction (scaled by depth for parallax)
            const dx = mouseX - this.x;
            const dy = mouseY - this.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < CONFIG.mouseInfluence) {
                const force = (1 - dist / CONFIG.mouseInfluence) * CONFIG.mouseForce * this.depth;
                this.vx += dx / dist * force;
                this.vy += dy / dist * force;
            }

            // Apply friction
            this.vx *= CONFIG.friction;
            this.vy *= CONFIG.friction;

            // Update position
            this.x += this.vx;
            this.y += this.vy;

            // Wrap around screen edges
            if (this.x < -20) this.x = width + 20;
            if (this.x > width + 20) this.x = -20;
            if (this.y < -20) this.y = height + 20;
            if (this.y > height + 20) this.y = -20;
        }

        draw() {
            const colorBase = CONFIG.colors[this.colorIndex];
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size * this.depth, 0, Math.PI * 2);
            ctx.fillStyle = colorBase + (this.alpha * this.depth) + ')';
            ctx.fill();
        }
    }

    // ---- Initialize Particles ----
    function init() {
        resize();
        particles = [];
        for (let i = 0; i < CONFIG.particleCount; i++) {
            particles.push(new Particle());
        }
    }

    // ---- Draw Connections ----
    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < CONFIG.connectDistance) {
                    const opacity = (1 - dist / CONFIG.connectDistance) * CONFIG.lineOpacity;
                    const avgDepth = (particles[i].depth + particles[j].depth) / 2;

                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(120, 160, 255, ${opacity * avgDepth})`;
                    ctx.lineWidth = 0.6 * avgDepth;
                    ctx.stroke();
                }
            }
        }
    }

    // ---- Mouse Glow Effect ----
    function drawMouseGlow() {
        if (mouseX < 0 || mouseY < 0) return;

        const gradient = ctx.createRadialGradient(
            mouseX, mouseY, 0,
            mouseX, mouseY, CONFIG.glowBlur
        );
        gradient.addColorStop(0, 'rgba(0, 229, 255, 0.06)');
        gradient.addColorStop(0.5, 'rgba(179, 136, 255, 0.02)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
    }

    // ---- Animation Loop ----
    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Draw glow behind particles
        drawMouseGlow();

        // Update and draw particles
        for (const p of particles) {
            p.update();
            p.draw();
        }

        // Draw connections
        drawConnections();

        requestAnimationFrame(animate);
    }

    // ---- Event Listeners ----
    window.addEventListener('resize', () => {
        resize();
        // Reposition particles that are out of bounds
        for (const p of particles) {
            if (p.x > width) p.x = Math.random() * width;
            if (p.y > height) p.y = Math.random() * height;
        }
    });

    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    window.addEventListener('mouseleave', () => {
        mouseX = -1000;
        mouseY = -1000;
    });

    // Touch support for mobile
    window.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) {
            mouseX = e.touches[0].clientX;
            mouseY = e.touches[0].clientY;
        }
    }, { passive: true });

    window.addEventListener('touchend', () => {
        mouseX = -1000;
        mouseY = -1000;
    });

    // ---- Start ----
    init();
    animate();
})();
