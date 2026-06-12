# Frontend Blank Page Debug Fixes

## Issues Found

### 1. WebSocket Race Condition (Backend)
**Problem:** `"WebSocket is not connected. Need to call 'accept' first"` error  
**Location:** `python-backend/src/server/connection_manager.py`  
**Root Cause:** The `send()` method tried to disconnect a WebSocket that was already disconnected, causing attempts to call methods on closed connections.

**Fix Applied:**
```python
async def send(self, ws: WebSocket, message: str) -> None:
    """Send a message to a single client. Disconnects on failure."""
    # Check if websocket is still in our active connections
    async with self._lock:
        if ws not in self._connections:
            return  # Already disconnected, skip silently
    
    try:
        await ws.send_text(message)
    except Exception:
        # Only disconnect if still in our set (avoid double-disconnect)
        async with self._lock:
            self._connections.discard(ws)
```

### 2. Frontend Connection Timeout Too Short
**Problem:** 3-second timeout was too aggressive for slower systems or during backend startup  
**Location:** `src/hooks/useWebSocket.ts`

**Fix Applied:**
- Increased timeout from 3s to 5s
- Added detailed connection close logging with error codes
- Added error event logging for better debugging

### 3. No Visual Feedback for Connection States
**Problem:** Blank page on refresh with no indication of what's happening  
**Location:** `src/App.tsx`

**Fix Applied:**
- Added "Connecting to Backend" overlay when `connectionStatus === 'connecting'`
- Added "Backend Offline" overlay when backend cannot be reached
- Includes instructions for starting Python backend
- Shows retry countdown

## Files Modified

### Backend:
1. `python-backend/src/server/connection_manager.py`
   - Fixed race condition in `send()` method
   - Added connection state check before sending

### Frontend:
1. `src/hooks/useWebSocket.ts`
   - Increased connection timeout from 3s to 5s
   - Added detailed WebSocket close event logging
   - Added error event logging
   - Improved reconnection logic with console messages

2. `src/App.tsx`
   - Added "Connecting to Backend" overlay
   - Added "Backend Offline" overlay with instructions
   - Better UX for connection states

## Testing the Fixes

### Scenario 1: Normal Connection
1. Start Python backend: `cd python-backend && python main.py`
2. Start frontend: `npm run dev`
3. **Expected:** Frontend connects within 5 seconds, shows normal UI

### Scenario 2: Backend Offline
1. Stop Python backend (if running)
2. Refresh frontend
3. **Expected:** "Backend Offline" overlay appears with instructions

### Scenario 3: Backend Restart
1. With frontend open, stop Python backend
2. Restart Python backend
3. **Expected:** Frontend automatically reconnects within 3 seconds

### Scenario 4: Page Refresh During Active Session
1. Start both backend and frontend
2. Start pipeline
3. Refresh page
4. **Expected:** Smooth reconnection, pipeline state restored

## Error Messages Fixed

### Before:
```
[WS] Unexpected connection error: WebSocket is not connected. Need to call "accept" first.
[WS] Client disconnected (active: 0)
```

### After:
```
[WS] Connection closed: code=1000, reason=none, clean=true
[WS] Client disconnected (active: 0)
```

## Connection Flow

### Successful Connection:
1. Frontend: Health check → backend online
2. Frontend: WebSocket connect attempt
3. Backend: Accept WebSocket connection
4. Backend: Add to connection manager
5. Backend: Send initial status, device list, model list
6. Frontend: Receive data → render UI
7. ✅ **Success:** No blank page

### Failed Connection (Now Handled):
1. Frontend: Health check → backend offline
2. Frontend: Show "Backend Offline" overlay
3. Frontend: Poll health every 3s
4. When backend comes online → auto-connect
5. ✅ **Success:** Clear feedback, no confusion

## Additional Improvements

### Console Logging:
- WebSocket close codes and reasons now logged
- Reconnection attempts now visible in console
- Better debugging for connection issues

### User Experience:
- No more blank pages
- Clear visual feedback for all connection states
- Instructions for fixing offline backend
- Automatic reconnection without user action

## Performance Impact

- **Connection timeout:** 3s → 5s (safer for slower systems)
- **Reconnection polling:** Still 3s (unchanged)
- **Stale connection timeout:** 30s (unchanged)
- **Memory:** Minimal overhead from overlay components

## Browser Compatibility

All fixes use standard Web APIs:
- WebSocket API (all modern browsers)
- Console logging (all browsers)
- React overlays (all browsers with React support)

## Future Improvements

Consider adding:
1. Exponential backoff for reconnection attempts
2. Manual reconnect button on offline overlay
3. Network status detection (online/offline events)
4. Connection quality indicator (latency, dropped frames)
5. Configurable connection timeout in settings
