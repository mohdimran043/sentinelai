import { setupServer } from 'msw/node'
import { eventHandlers } from '@/events/mocks/handlers'

export const server = setupServer(...eventHandlers)
