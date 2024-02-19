/* import { toggleOverlay } from './scripts/loader.js'
import { overlay } from './scripts/overlay.js'
import { logoComponent } from './scripts/logo-component.js'
import { onNavPopup, onSidebarPopup } from './scripts/popup.js'

overlay('overlay')
toggleOverlay('overlay')
logoComponent('logo-component')
onNavPopup('nav')

if (document.getElementById('details')) {onSidebarPopup('#details')} */



import { toggleOverlay } from './loader.js'
import { overlay } from './overlay.js'
import { logoComponent } from './logo-component.js'
import { onNavPopup, onSidebarPopup } from './popup.js'

overlay('overlay')
toggleOverlay('overlay')
logoComponent('logo-component')
onNavPopup('nav')

if (document.getElementById('details')) {onSidebarPopup('#details')}