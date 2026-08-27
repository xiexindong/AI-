var b = {
  'id': '987a6618-23c3-4077-8f41-e5255303570a',
  'object': 'chat.completion',
  'created': 1787825686,
  'model': 'deepseek-v4-flash',
  'choices': [{
    'index': 0,
    'message': {
      'role': 'assistant',
      'content': '',
      'tool_calls': [{
        'index': 0,
        'id': 'call_00_godORmL0ELhdVoOYBhQy1988',
        'type': 'function',
        'function': {
          'name': 'get_current_time',
          'arguments': '{}'
        }
      }]
    },
    'logprobs': None,
    'finish_reason': 'tool_calls'
  }],
  'usage': {
    'prompt_tokens': 334,
    'completion_tokens': 28,
    'total_tokens': 362,
    'prompt_tokens_details': {
      'cached_tokens': 256
    },
    'prompt_cache_hit_tokens': 256,
    'prompt_cache_miss_tokens': 78
  },
  'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e'
}

var a = {
  'role': 'assistant',
  'content': '',
  'tool_calls': [{
    'index': 0,
    'id': 'call_00_godORmL0ELhdVoOYBhQy1988',
    'type': 'function',
    'function': {
      'name': 'get_current_time',
      'arguments': '{}'
    }
  }]
}
