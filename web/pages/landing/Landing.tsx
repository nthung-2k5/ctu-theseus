import {
  ArchiveIcon,
  BrainIcon,
  CaretDownIcon,
  ChartLineIcon,
  ExportIcon,
  FlaskIcon,
  PlayIcon,
  StackIcon,
  TagIcon,
  UploadSimpleIcon,
} from '@phosphor-icons/react'
import { LinkButton } from '@public/components/ui'
import { type CSSProperties, type ReactNode, useEffect, useRef } from 'react'
import { LiveCard } from './LiveCard'
import { NeuralCanvas } from './NeuralCanvas'
import { CountUp, Reveal } from './Reveal'
import './landing.css'

const HEADLINE = ['Train', 'models', 'you', 'can', 'watch', 'learn.']

const STEPS = [
  {
    icon: UploadSimpleIcon,
    title: 'Upload',
    text: 'Drop folders, archives, CSVs or text. Class and split folders are detected for you.',
  },
  {
    icon: TagIcon,
    title: 'Classes',
    text: 'Define and colour your label classes, or let the folder layout create them.',
  },
  {
    icon: ArchiveIcon,
    title: 'Snapshot',
    text: 'Split, augment the training set and freeze it as an immutable, versioned snapshot.',
  },
  {
    icon: BrainIcon,
    title: 'Train',
    text: 'Launch runs or hyper-parameter sweeps and watch curves and logs stream live.',
  },
  {
    icon: ExportIcon,
    title: 'Export',
    text: 'Test in the playground, then export ONNX or another format and call the prediction API.',
  },
]

const FEATURES = [
  {
    icon: StackIcon,
    title: 'One project, one task',
    text: 'Every project is a single problem: one modality, one task. Datasets, runs and exports never get mixed up.',
  },
  {
    icon: FlaskIcon,
    title: 'Pluggable trainers',
    text: 'Training backends are plugins. Pick a backend, a model and its hyper-parameters straight from the server.',
  },
  {
    icon: ArchiveIcon,
    title: 'Snapshots you can open',
    text: 'Each snapshot is immutable. Browse its items, classes and split at any time.',
  },
  {
    icon: ChartLineIcon,
    title: 'Augmentation, baked in',
    text: 'Multiply or balance classes with modality-aware augmentations, applied to the train split only so nothing leaks.',
  },
  {
    icon: TagIcon,
    title: 'Classes you control',
    text: 'For classification tasks, add, rename and colour classes. Regression projects simply have none.',
  },
  {
    icon: PlayIcon,
    title: 'Playground and export',
    text: 'Try a trained model on real inputs, then package it in the format you deploy with.',
  },
]

/** Cursor-following spotlight on feature cards (sets CSS variables; no re-render). */
function Spotlight({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  return (
    <div
      ref={ref}
      className="ot-card ot-feature"
      onPointerMove={(e) => {
        const el = ref.current
        if (!el) return
        const r = el.getBoundingClientRect()
        el.style.setProperty('--mx', `${e.clientX - r.left}px`)
        el.style.setProperty('--my', `${e.clientY - r.top}px`)
      }}
    >
      {children}
    </div>
  )
}

export function Landing() {
  // Landing owns its scroll container.
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    root.current?.focus({ preventScroll: true })
  }, [])

  return (
    <div className="ot-landing" ref={root} tabIndex={-1}>
      <header className="ot-nav">
        <div className="ot-logo">
          <span className="ot-logo-mark" /> CTU Theseus
        </div>
        <nav className="ot-nav-links" aria-label="Sections">
          <a href="#how">How it works</a>
          <a href="#features">Features</a>
        </nav>
        <div className="ot-nav-cta">
          <LinkButton to="/login" search={{ mode: 'signin' }} variant="subtle" color="gray">
            Sign in
          </LinkButton>
          <LinkButton to="/login" search={{ mode: 'register' }}>
            Get started
          </LinkButton>
        </div>
      </header>

      {/* ------------------------------------------------------------ hero */}
      <section className="ot-hero">
        <NeuralCanvas />
        <div className="ot-aurora" aria-hidden="true">
          <i />
          <i />
          <i />
        </div>
        <div className="ot-hero-inner">
          <div className="ot-hero-copy">
            <div className="ot-pill">
              <span className="ot-live-dot" /> ML training platform
            </div>
            <h1 className="ot-h1" aria-label={HEADLINE.join(' ')}>
              {HEADLINE.map((w, i) => (
                <span
                  key={w}
                  className={`ot-word ${i >= 4 ? 'ot-grad' : ''}`}
                  style={{ animationDelay: `${120 + i * 90}ms` }}
                  aria-hidden="true"
                >
                  {w}&nbsp;
                </span>
              ))}
            </h1>
            <p className="ot-lead">
              Curate datasets, snapshot and augment them, configure and launch training runs, and watch them happen in
              real time, then test and export the result.
            </p>
            <div className="ot-cta-row">
              <LinkButton to="/login" search={{ mode: 'register' }} size="md" className="ot-glow">
                Create your workspace
              </LinkButton>
              <LinkButton to="/login" search={{ mode: 'signin' }} variant="default" size="md">
                Sign in
              </LinkButton>
            </div>
            <div className="ot-trust">
              <span>Vision</span>
              <span>Text</span>
              <span>Audio</span>
              <span>Tabular</span>
            </div>
          </div>
          <div className="ot-hero-visual">
            <LiveCard />
          </div>
        </div>
        <a href="#how" className="ot-scroll" aria-label="Scroll to how it works">
          <CaretDownIcon size={22} />
        </a>
      </section>

      {/* ------------------------------------------------------------ pipeline */}
      <section id="how" className="ot-section">
        <Reveal>
          <h2 className="ot-h2">From raw data to a shipped model</h2>
        </Reveal>
        <Reveal delay={80}>
          <p className="ot-sub">One workflow, five steps, no context switching.</p>
        </Reveal>
        <Reveal className="ot-pipeline" delay={120}>
          <div className="ot-rail">
            <div className="ot-rail-fill" />
            <div className="ot-packet" />
          </div>
          <ol>
            {STEPS.map((s, i) => (
              <li key={s.title} style={{ '--i': i } as CSSProperties}>
                <span className="ot-step-icon">
                  <s.icon size={22} />
                </span>
                <b>{s.title}</b>
                <p>{s.text}</p>
              </li>
            ))}
          </ol>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------ features */}
      <section id="features" className="ot-section">
        <Reveal>
          <h2 className="ot-h2">Built for how research actually goes</h2>
        </Reveal>
        <div className="ot-grid">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} delay={(i % 3) * 90}>
              <Spotlight>
                <span className="ot-feature-icon">
                  <f.icon size={22} />
                </span>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
              </Spotlight>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ stats */}
      <section className="ot-stats">
        <Reveal className="ot-stat">
          <b>
            <CountUp to={4} />
          </b>
          <span>modalities, one workflow</span>
        </Reveal>
        <Reveal className="ot-stat" delay={80}>
          <b>
            <CountUp to={5} />
          </b>
          <span>steps from upload to export</span>
        </Reveal>
        <Reveal className="ot-stat" delay={160}>
          <b>
            <CountUp to={100} suffix="%" />
          </b>
          <span>immutable, reproducible snapshots</span>
        </Reveal>
        <Reveal className="ot-stat" delay={240}>
          <b>
            <CountUp to={1} />
          </b>
          <span>API for every trained model</span>
        </Reveal>
      </section>

      {/* ------------------------------------------------------------ cta */}
      <section className="ot-section ot-final">
        <div className="ot-aurora ot-aurora-soft" aria-hidden="true">
          <i />
          <i />
          <i />
        </div>
        <Reveal>
          <h2 className="ot-h2">Start your first project</h2>
        </Reveal>
        <Reveal delay={80}>
          <p className="ot-sub">Pick a task, drop in your data, and be training in minutes.</p>
        </Reveal>
        <Reveal delay={160}>
          <div className="ot-cta-row ot-center">
            <LinkButton to="/login" search={{ mode: 'register' }} size="lg" className="ot-glow">
              Create account
            </LinkButton>
            <LinkButton to="/login" search={{ mode: 'signin' }} variant="default" size="lg">
              I already have one
            </LinkButton>
          </div>
        </Reveal>
      </section>

      <footer className="ot-footer">CTU Theseus · train, evaluate and export models</footer>
    </div>
  )
}
