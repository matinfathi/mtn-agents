from typing import Union, TypeVar, TypeAlias, Dict, List, TypedDict, Optional, Generator
from typing_extensions import override, Literal

_T = TypeVar("_T")


class NotGiven:
    
    def __bool__(self) -> Literal[False]:
        return False
    
    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = Union[_T, NotGiven]
NOT_GIVEN = NotGiven()

ProductDescriptionDict: TypeAlias = Dict[str, Union[str, Dict[str, str]]]

ReferenceList: TypeAlias = List[Dict[str, str]]

ResponseMetadataDict: TypeAlias = Dict[str, Union[str, ReferenceList]]

AgentResponseDict: TypeAlias = Dict[str, Union[str, ResponseMetadataDict]]


class FilterDict(TypedDict, total=False):
    document_id: Dict[str, List[str]]
    cache_id: Optional[Dict[str, Union[List[str], bool]]]


ChainResponse = Dict[str, Union[str, Dict[str, str], Generator[str, None, None]]]

OpenAITools = List[Dict[
    str, Union[str, Dict[str, Union[str, Dict[str, Union[str, Dict[str, Union[str, Dict[str, Dict[str, str]]]]]]]]]]]
