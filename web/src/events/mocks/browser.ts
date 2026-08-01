import { setupWorker } from 'msw/browser'
import { eventHandlers } from '@/events/mocks/handlers'

export const worker = setupWorker(...eventHandlers)
