import { mount } from 'svelte';
import '@xyflow/svelte/dist/style.css';
import './diagram.css';
import Diagram from './Diagram.svelte';

mount(Diagram, { target: document.getElementById('flow') });
