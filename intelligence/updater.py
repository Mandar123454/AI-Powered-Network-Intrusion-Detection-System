"""
Feed Updater for AI-NIDS

This module provides:
- Scheduled feed updates
- Background update tasks
- Update status tracking
- Failure handling and retry logic

Author: AI-NIDS Team
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading

from .ioc_feeds import FeedManager, IOCFeed

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Status of a feed update."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class UpdateResult:
    """Result of a feed update operation."""
    feed_name: str
    status: UpdateStatus
    entries_updated: int
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feed_name': self.feed_name,
            'status': self.status.value,
            'entries_updated': self.entries_updated,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'duration_seconds': self.duration_seconds
        }


@dataclass
class ScheduledTask:
    """A scheduled update task."""
    feed_name: str
    interval: timedelta
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))


class FeedUpdater:
    """
    Manages feed updates with retry logic.
    
    Features:
    - Automatic retry on failure
    - Exponential backoff
    - Update history tracking
    - Concurrent update support
    """
    
    def __init__(
        self,
        feed_manager: FeedManager,
        max_concurrent_updates: int = 3,
        max_retries: int = 3,
        base_retry_delay: int = 60  # seconds
    ):
        self.feed_manager = feed_manager
        self.max_concurrent = max_concurrent_updates
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        
        # Update history
        self._update_history: List[UpdateResult] = []
        self._max_history = 1000
        
        # Callbacks
        self._on_success_callbacks: List[Callable] = []
        self._on_failure_callbacks: List[Callable] = []
    
    def on_success(self, callback: Callable):
        """Register callback for successful updates."""
        self._on_success_callbacks.append(callback)
    
    def on_failure(self, callback: Callable):
        """Register callback for failed updates."""
        self._on_failure_callbacks.append(callback)
    
    async def update_feed(
        self,
        feed_name: str,
        force: bool = False
    ) -> UpdateResult:
        """
        Update a single feed.
        
        Args:
            feed_name: Name of the feed to update
            force: Force update even if not due
        
        Returns:
            UpdateResult with status and details
        """
        if feed_name not in self.feed_manager.feeds:
            return UpdateResult(
                feed_name=feed_name,
                status=UpdateStatus.FAILED,
                entries_updated=0,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error_message=f"Feed '{feed_name}' not found"
            )
        
        feed = self.feed_manager.feeds[feed_name]
        
        if not force and not feed.needs_update():
            return UpdateResult(
                feed_name=feed_name,
                status=UpdateStatus.SKIPPED,
                entries_updated=0,
                started_at=datetime.now(),
                completed_at=datetime.now()
            )
        
        started_at = datetime.now()
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Updating feed '{feed_name}' (attempt {attempt + 1})")
                
                entries_count = await feed.update()
                
                completed_at = datetime.now()
                duration = (completed_at - started_at).total_seconds()
                
                result = UpdateResult(
                    feed_name=feed_name,
                    status=UpdateStatus.SUCCESS,
                    entries_updated=entries_count,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration
                )
                
                self._add_to_history(result)
                await self._trigger_success_callbacks(result)
                
                logger.info(f"Feed '{feed_name}' updated: {entries_count} entries in {duration:.2f}s")
                return result
                
            except Exception as e:
                logger.warning(f"Feed '{feed_name}' update failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.base_retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    completed_at = datetime.now()
                    duration = (completed_at - started_at).total_seconds()
                    
                    result = UpdateResult(
                        feed_name=feed_name,
                        status=UpdateStatus.FAILED,
                        entries_updated=0,
                        started_at=started_at,
                        completed_at=completed_at,
                        error_message=str(e),
                        duration_seconds=duration
                    )
                    
                    self._add_to_history(result)
                    await self._trigger_failure_callbacks(result)
                    
                    return result
        
        # Should not reach here
        return UpdateResult(
            feed_name=feed_name,
            status=UpdateStatus.FAILED,
            entries_updated=0,
            started_at=started_at,
            completed_at=datetime.now(),
            error_message="Unknown error"
        )
    
    async def update_all_feeds(
        self,
        force: bool = False
    ) -> List[UpdateResult]:
        """
        Update all feeds concurrently.
        
        Args:
            force: Force update even if not due
        
        Returns:
            List of UpdateResults for all feeds
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def update_with_limit(feed_name: str) -> UpdateResult:
            async with semaphore:
                return await self.update_feed(feed_name, force)
        
        tasks = [
            update_with_limit(name)
            for name in self.feed_manager.feeds.keys()
        ]
        
        return await asyncio.gather(*tasks)
    
    def _add_to_history(self, result: UpdateResult):
        """Add result to history, maintaining max size."""
        self._update_history.append(result)
        if len(self._update_history) > self._max_history:
            self._update_history = self._update_history[-self._max_history:]
    
    async def _trigger_success_callbacks(self, result: UpdateResult):
        """Trigger success callbacks."""
        for callback in self._on_success_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Success callback error: {e}")
    
    async def _trigger_failure_callbacks(self, result: UpdateResult):
        """Trigger failure callbacks."""
        for callback in self._on_failure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def get_history(
        self,
        feed_name: Optional[str] = None,
        status: Optional[UpdateStatus] = None,
        limit: int = 100
    ) -> List[UpdateResult]:
        """Get update history with optional filters."""
        results = self._update_history
        
        if feed_name:
            results = [r for r in results if r.feed_name == feed_name]
        
        if status:
            results = [r for r in results if r.status == status]
        
        return results[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get updater statistics."""
        total = len(self._update_history)
        by_status = {}
        for status in UpdateStatus:
            count = sum(1 for r in self._update_history if r.status == status)
            by_status[status.value] = count
        
        # Calculate average duration for successful updates
        successful = [r for r in self._update_history if r.status == UpdateStatus.SUCCESS]
        avg_duration = (
            sum(r.duration_seconds for r in successful) / len(successful)
            if successful else 0
        )
        
        return {
            'total_updates': total,
            'by_status': by_status,
            'average_duration_seconds': avg_duration,
            'feeds_count': len(self.feed_manager.feeds)
        }


class UpdateScheduler:
    """
    Scheduler for automatic feed updates.
    
    Features:
    - Configurable intervals per feed
    - Background update task
    - Pause/resume support
    - Update time optimization
    """
    
    def __init__(
        self,
        updater: FeedUpdater,
        default_interval_minutes: int = 60
    ):
        self.updater = updater
        self.default_interval = timedelta(minutes=default_interval_minutes)
        
        # Scheduled tasks
        self._tasks: Dict[str, ScheduledTask] = {}
        
        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
    
    def schedule_feed(
        self,
        feed_name: str,
        interval_minutes: Optional[int] = None,
        enabled: bool = True
    ):
        """Schedule a feed for regular updates."""
        interval = (
            timedelta(minutes=interval_minutes)
            if interval_minutes
            else self.default_interval
        )
        
        self._tasks[feed_name] = ScheduledTask(
            feed_name=feed_name,
            interval=interval,
            next_run=datetime.now() + interval,
            enabled=enabled
        )
    
    def unschedule_feed(self, feed_name: str):
        """Remove a feed from the schedule."""
        if feed_name in self._tasks:
            del self._tasks[feed_name]
    
    def enable_feed(self, feed_name: str):
        """Enable updates for a feed."""
        if feed_name in self._tasks:
            self._tasks[feed_name].enabled = True
    
    def disable_feed(self, feed_name: str):
        """Disable updates for a feed."""
        if feed_name in self._tasks:
            self._tasks[feed_name].enabled = False
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        
        self._running = True
        
        # Schedule all known feeds
        for feed_name in self.updater.feed_manager.feeds.keys():
            if feed_name not in self._tasks:
                self.schedule_feed(feed_name)
        
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Update scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Update scheduler stopped")
    
    def pause(self):
        """Pause the scheduler."""
        self._pause_event.clear()
        logger.info("Update scheduler paused")
    
    def resume(self):
        """Resume the scheduler."""
        self._pause_event.set()
        logger.info("Update scheduler resumed")
    
    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Wait if paused
                await self._pause_event.wait()
                
                # Find tasks due for update
                now = datetime.now()
                due_tasks = [
                    task for task in self._tasks.values()
                    if task.enabled and task.next_run and task.next_run <= now
                ]
                
                # Run due updates
                for task in due_tasks:
                    result = await self.updater.update_feed(task.feed_name)
                    
                    # Update task state
                    task.last_run = now
                    task.next_run = now + task.interval
                    
                    if result.status == UpdateStatus.FAILED:
                        task.retry_count += 1
                    else:
                        task.retry_count = 0
                
                # Sleep until next check (1 minute)
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def get_schedule(self) -> List[Dict[str, Any]]:
        """Get the current schedule."""
        return [
            {
                'feed_name': task.feed_name,
                'interval_minutes': task.interval.total_seconds() / 60,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'enabled': task.enabled,
                'retry_count': task.retry_count
            }
            for task in self._tasks.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'running': self._running,
            'paused': not self._pause_event.is_set(),
            'scheduled_feeds': len(self._tasks),
            'enabled_feeds': sum(1 for t in self._tasks.values() if t.enabled)
        }


def create_updater(
    feed_manager: FeedManager,
    max_concurrent: int = 3,
    max_retries: int = 3
) -> FeedUpdater:
    """Create a feed updater."""
    return FeedUpdater(
        feed_manager=feed_manager,
        max_concurrent_updates=max_concurrent,
        max_retries=max_retries
    )
